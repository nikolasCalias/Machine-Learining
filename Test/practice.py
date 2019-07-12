import random
roundsPlayed = 0
playerOneWin = 0
playerTwoWin = 0
roundMax = int(input("How many rounds would you like to play: "))
while roundsPlayed < roundMax:  # Game ends when the rounds played exceeds the round maximum
    for x in range(1):
        playerTwo = random.randint(0, 2)  # 0 is rock, 1 is paper, 2 is scissors
    playerOne = input("Player, choose rock, paper, or scissors: ")
# converts the player's input into integers for later comparison. 0 is rock, 1 is paper, and 2 is scissors
    if playerOne == "rock":
        playerOne = 0
    elif playerOne == "paper":
        playerOne = 1
    else:
        playerOne = 2
    roundsPlayed += 1
    if playerOne == 0:  # The win and lose conditions when P1 chooses rock
        if playerTwo == 2:
            print("You win, rock beats scissors")
            playerOneWin += 1
        elif playerTwo == 1:
            print("You lose, paper beats rock")
            playerTwoWin += 1
        else:
            print("Tie game!")
    elif playerOne == 1:  # win and lose conditions when P1 chooses paper
        if playerTwo == 0:
            print("You win, paper beats rock")
            playerOneWin += 1
        elif playerTwo == 2:
            print("You lose, scissors beats paper")
            playerTwoWin += 1
        else:
            print("Tie game!")
    elif playerOne == 2:  # Win and lose conditions when P1 chooses scissors
        if playerTwo == 1:
            print("You win, scissors beats paper")
            playerOneWin += 1
        elif playerTwo == 0:
            print("You lose,rock beats scissors")
            playerTwoWin += 1
        else:
            print("Tie game!")
    print("The current score is You: %d and PC: %d" % (playerOneWin, playerTwoWin))
if playerOneWin > playerTwoWin:
    print("You win!%d to %d" % (playerOneWin, playerTwoWin))  # Player1 win message
elif playerTwoWin > playerOneWin:
    print("You've lost! %d to %d" % (playerTwoWin, playerOneWin))  # Computer's win message
else:
    print("Tie game! Everyone's a winner!")  # Tie game message
