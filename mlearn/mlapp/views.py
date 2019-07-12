from django.shortcuts import render
from django.views.generic import TemplateView
from mlapp.models import DownloadedFile, CurrentFile, Prepross
from static.py.funcs import StrToList
import pandas as pd
import os
import plotly.offline as opy
import plotly.graph_objs as go
from sklearn import metrics
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from xgboost import XGBRegressor


class index(TemplateView):
    template_name = 'index.html'

    def post(self, request, **kwargs):
        if request.method == 'POST':
            current_file = CurrentFile(filename=list(request.POST.keys())[1])
            current_file.save()
            context = {}
            context['fileselected'] = list(request.POST.keys())[1]
            context['test'] = 'a'
            return render(request, 'index.html', context)

    def __str__(self):
        return self.name


class upload(TemplateView):
    template_name = 'upload.html'

    def post(self, request, **kwargs):
        if request.method == 'POST' and request.FILES["docfile"]:

            downloaded_file = DownloadedFile(docfile=request.FILES['docfile'])
            downloaded_file.save()
            downloaded_file.delete() # it deletes only from database
            return render(request,'upload.html')

    def __str__(self):
        return self.name


class splitting(TemplateView):
    template_name = 'splitting.html'

    def post(self, request, **kwargs):
        if request.method == 'POST':
            current_file = CurrentFile(filename = request.POST.getlist('recipients[]'))
            current_file.save()
            name = request.POST.get('name')
            if name == 'train':
                context1 = {}
                context1['fileselected1'] = request.POST.getlist('recipients[]')
                return render(request,'splitting.html', context1)
            elif name == 'test':
                context2 = {}
                context2['fileselected2'] = request.POST.getlist('recipients[]')
                return render(request, 'splitting.html', context2)

    def __str__(self):
        return self.name


class preprocessing(TemplateView):
    template_name = 'preprocessing.html'
    #model = Prepross
    #fields = ('file_name','coltype','Xvars','Yvar','onehot','featscaling','na_omit')

    def get_context_data(self, **kwargs):
        data = super().get_context_data(**kwargs)
        context = {}
        try:

            file_name = CurrentFile.objects.order_by('-id')[1].filename
            file_name = file_name.replace('{', '')
            file_name = file_name.replace('}', '')
            file_name = [x.strip() for x in file_name.split(',')]
            df = pd.concat([pd.read_csv(os.path.join('media\downloaded', file)) for file in file_name],)
            count_nan = len(df) - df.count()
            row_count = df.count()[0]
            file_type = pd.concat([df.dtypes, count_nan, df.nunique()], axis=1)
            file_type.columns = ("Type", "NA count", "Count distinct")

            context['file_type'] = file_type
            context['row_count'] = row_count
        except:
            file_name = 'Please select one file'


        context['file_name'] = file_name
        return context

    def post(self, request, **kwargs):
        if request.method == 'POST':

            # Entry to model
            file_name = CurrentFile.objects.order_by('-id')[1].filename
            file_name = file_name.replace('{', '')
            file_name = file_name.replace('}', '')
            file_name = [x.strip() for x in file_name.split(',')]

            prep_file = Prepross.objects.get_or_create(filename = file_name[0],
                                  coltype = request.POST.getlist('coltype'),
                                    assvar = request.POST.getlist('assvar'),
                            missingvalues = request.POST.getlist('missingvalues'),
                         trainingset_size = request.POST['trainingset_size'],
                              featscaling = request.POST.getlist('featscaling')
                                   )


            # Get dataframe and change data type
            context = {}

            coltype = request.POST.getlist('coltype')
            coltype = dict([i.split(':', 1) for i in coltype])
            df = pd.concat([pd.read_csv(os.path.join('media\downloaded', file)) for file in file_name],)
            row_count = df.count()[0]

            # Keep only selected columns
            assvar = request.POST.getlist('assvar')
            xcols0 = [s for s in assvar if ":X" in s]
            xcols = [i.split(':', 1)[0] for i in xcols0]
            ycol0 = [s for s in assvar if ":y" in s]
            ycol = [i.split(':', 1)[0] for i in ycol0]
            cols = xcols + ycol
            df = df[cols]

            xcols = ', '.join(xcols)
            ycol = ', '.join(ycol)
            missing = request.POST.getlist('missingvalues')
            missing = ', '.join(missing)
            trainingset_s = request.POST.getlist('trainingset_size')
            trainingset_s = ', '.join(trainingset_s)
            testset_s = 100 - int(trainingset_s)
            feat =  request.POST['featscaling']

            # Taking care of missing data
            if missing == "no":
                if len(df) != len(df.dropna()):
                    context['selecty'] = 'Your data seem to have Missing Values'
                else:
                    df = df.dropna()


            # Return error if columns are not selected
            if len(ycol0) != 1:
                context['selecty'] = 'Please select one y variable'

            elif len(xcols0) < 1:
                context['selecty'] = 'Please select one or more X variables'

            else:
            # Plot
                graph = {}
                for i in df.columns:
                    layout = go.Layout(autosize=False, width=400, height=400,
                        title= i,
                        xaxis=dict(title='Value'),
                        yaxis=dict(title='Count'),
                        bargap=0.2,
                        bargroupgap=0.1)
                    data = go.Figure(data=[go.Histogram(x=df[i])], layout=layout)
                    graph[i] = opy.plot(data, include_plotlyjs=False, output_type='div')
                context['graph'] = graph

            context['xcols'] = xcols
            context['ycol'] = ycol
            context['missing'] = missing
            context['trainingset_s'] = trainingset_s
            context['testset_s'] = testset_s
            context['feat'] = feat
            context['file_name'] = file_name
            context['row_count'] = row_count
        return render(request,'preprocessing.html', context)

    def __str__(self):
        return self.name


class modelling(TemplateView):
    template_name = 'modelling.html'

    def get_context_data(self, **kwargs):
        data = super().get_context_data(**kwargs)
        context = {}
        try:
            # filter avoids csv file (CurrentFile ansd Prepross model) to be different from preprocessing parameters

            file_name = CurrentFile.objects.order_by('-id')[0].filename
            file_name = file_name.replace('{', '')
            file_name = file_name.replace('}', '')
            file_name = [x.strip() for x in file_name.split(',')]
            df = pd.concat([pd.read_csv(os.path.join('media\downloaded', file)) for file in file_name], )
            prepross_dict = Prepross.objects.filter(filename = file_name[0]).order_by('-id').values()[0]
            #file_name = prepross_dict['filename']
            assvar = StrToList.strtolist(prepross_dict['assvar'])
            xcols = [s for s in assvar if ":X" in s]
            xcols = [i.split(':', 1)[0] for i in xcols]
            ycol = [s for s in assvar if ":y" in s]
            ycol = [i.split(':', 1)[0] for i in ycol]

            xcols = ', '.join(xcols)
            ycol = ', '.join(ycol)
            missing = StrToList.strtolist(prepross_dict['missingvalues'])
            missing = ', '.join(missing)
            trainingset_s = prepross_dict['trainingset_size']
            testset_s = 100 - prepross_dict['trainingset_size']
            feat =  StrToList.strtolist(prepross_dict['featscaling'])
            feat= ', '.join(feat)
            context['xcols'] = xcols
            context['ycol'] = ycol
            context['missing'] = missing
            context['trainingset_s'] = trainingset_s
            context['testset_s'] = testset_s
            context['feat'] = feat
            context['file_name'] = file_name

        except:
            prepross_dict = 'Please submit preprocessing'

        context['prepross_dict'] = prepross_dict

        return context

    def post(self, request, **kwargs):
        if request.method == 'POST':

            file_name = CurrentFile.objects.order_by('-id')[0].filename
            file_name_test = CurrentFile.objects.order_by('-id')[1].filename
            file_name_test = file_name_test.replace('{', '')
            file_name_test = file_name_test.replace('}', '')
            file_name_test = [x.strip() for x in file_name_test.split(',')]

            file_name = file_name.replace('{', '')
            file_name = file_name.replace('}', '')
            file_name = [x.strip() for x in file_name.split(',')]
            context = {}
            # filter avoids csv file (CurrentFile ansd Prepross model) to be different from preprocessing parameters
            prepross_dict = Prepross.objects.filter(filename = file_name[0]).order_by('-id').values()[0]
            #file_name = prepross_dict['filename']

            # Get dataframe and change data type
            coltype = StrToList.strtolist(prepross_dict['coltype'])
            coltype = dict([i.split(':', 1) for i in coltype])
            #
            df = pd.concat([pd.read_csv(os.path.join('media\downloaded', file)) for file in file_name], )
            df_test = pd.concat([pd.read_csv(os.path.join('media\downloaded', file)) for file in file_name_test], )

            assvar = StrToList.strtolist(prepross_dict['assvar'])
            xcols = [s for s in assvar if ":X" in s]
            xcols = [i.split(':', 1)[0] for i in xcols]
            ycol = [s for s in assvar if ":y" in s]
            ycol = [i.split(':', 1)[0] for i in ycol]
            cols = xcols + ycol
            df = df[cols]
            df_test = df_test[cols]

            # Taking care of missing data
            if prepross_dict['missingvalues'] == "['na_omit']":
                df = df.dropna()


            X = df[xcols]
            X = pd.get_dummies(X, drop_first=True)
            X = X.values
            y = df[ycol]
            y = pd.get_dummies(y, drop_first=True)
            y = y.values

            X_test = df_test[xcols]
            X_test = pd.get_dummies(X_test, drop_first=True)
            X_test = X_test.values
            y_test = df_test[ycol]
            y_test = pd.get_dummies(y_test, drop_first=True)
            y_test = y_test.values

            # Splitting the dataset into the Training set and Test set
            testset_size = 1 - (prepross_dict['trainingset_size']/100)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testset_size)

            # Feature Scaling
            if prepross_dict['featscaling'] != "['no']":
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)

            p2 = {} # dictionary for plot
            accuracy = {} # dictionary for cm
            # Fit Classifications
            for k in request.POST.getlist('mlmodel'):
                if 'RF' == k:
                # Fitting Random Forest Classification to the Training set
                    classifier = RandomForestRegressor(n_estimators=1000, random_state=42
                                                       )
                    #classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
                elif 'logreg' == k:
                    # Fitting LogisticRegression to the Training set
                    classifier = LogisticRegression()
                elif 'xgboost' == k:
                    # Fitting xgboost to the Training set
                    classifier = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

                classifier.fit(X, y.ravel())

                # Predicting the Test set results
                y_pred = classifier.predict(X_test)

                y_pred
                # Making the Confusion Matrix
                #accuracy[k] = [k, round(metrics.accuracy_score(y_test, y_pred), 2)]
                #errors = abs(y_test.values - y_pred.values)
                errors = mean_squared_error(y_test, y_pred)

                r2 = r2_score(y_test, y_pred)

                # Calculating ROC curve
                #y_pred_proba = classifier.predict_proba(X_test)[::,1]
                #fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
                #auc = round(metrics.roc_auc_score(y_test, y_pred_proba), 2)

                # Plotting
             #   p2[k] = go.Scatter(x=fpr, y=tpr,
              #                  mode='lines',
               #                 name= k + ' auc=' + str(auc))

         #   p1 = go.Scatter(x=[0, 1], y=[0, 1],
          #                  mode='lines',
           #                 showlegend=False,
            #                line=dict(color='black', dash='dash')
             #              )

            #layout = go.Layout(title='ROC curve',
             #                  xaxis=dict(title='False positive rate'),
              #                 yaxis=dict(title='True positive rate')
               #               )
         #   fig = go.Figure(data=[p1] + list(p2.values()), layout=layout)
          #  graph = opy.plot(fig, include_plotlyjs=False, output_type='div')
           # context['graph'] = graph

            context['accuracy'] = errors
            context['r2'] = r2

            xcols = ', '.join(xcols)
            ycol = ', '.join(ycol)
            missing = StrToList.strtolist(prepross_dict['missingvalues'])
            missing = ', '.join(missing)
            trainingset_s = prepross_dict['trainingset_size']
            testset_s = 100 - prepross_dict['trainingset_size']
            feat =  StrToList.strtolist(prepross_dict['featscaling'])
            feat= ', '.join(feat)
            context['xcols'] = xcols
            context['ycol'] = ycol
            context['missing'] = missing
            context['trainingset_s'] = trainingset_s
            context['testset_s'] = testset_s
            context['feat'] = feat
            context['file_name'] = file_name

            return render(request, 'modelling.html', context)
