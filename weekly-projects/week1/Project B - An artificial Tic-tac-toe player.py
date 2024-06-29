import random

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 10)
        print(' ')

def check_winner(board, mark):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]],
    ]
    return [mark, mark, mark] in win_conditions

def check_block(board, mark):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]],
    ]
    return ([mark, mark, " "] in win_conditions) or ([" ", mark, mark] in win_conditions) or ([mark, " ", mark] in win_conditions)


def check_attack(board,mark):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]],
    ]
    return ([mark, " ", " "] in win_conditions) or ([" ", " ", mark] in win_conditions) or ([" ", mark, " "] in win_conditions)





def get_empty_positions(board):
    positions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                positions.append((i, j))
    return positions


def block(board):
    empty_positions = get_empty_positions(board)
    
    # Block user from winning
    for move in empty_positions:
        temp = [row[:] for row in board]
        temp[move[0]][move[1]] = "X"
        if check_winner(temp, "X"):
            board[move[0]][move[1]] = "O"
            return

def attack(board):
    empty_positions = get_empty_positions(board)
    
    # Place an attacking move
    for move in empty_positions:
        temp = [row[:] for row in board]
        temp[move[0]][move[1]] = "O"
        if check_block(temp, "O"):
            board[move[0]][move[1]] = "O"
            return

        
        
def winning_move(board):
    empty_positions = get_empty_positions(board)
    for move in empty_positions:
        temp = [row[:] for row in board]
        temp[move[0]][move[1]] = "O"
        if check_winner(temp, "O"):
            board[move[0]][move[1]] = "O"
            return

        
def user_move(board):
    while True:
        try:
            row = int(input("Enter the row (1-3): ")) - 1
            col = int(input("Enter the column (1-3): ")) - 1
            if board[row][col] == " ":
                board[row][col] = "X"
                break
            else:
                print("This position is already taken.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter numbers between 1 and 3.")

def computer_move(board, move):
    corners= [board[0][2], board[2][2], board[2][0],board[0][0]]
    
    #FIRST MOVE GO TO THE CORNERS
    if(move == 1):
        if("X" in corners):
            board[1][1] = "O"
        else:
            board[0][0] = "O"
                
    #SECOND MOVE. BLOCK IF HAVE TO OTHERWISE TAKE THE CORNERS
    elif(move == 2):
        
        if(check_block(board, "X")):
                block(board)
                
        #IF TWO CORNERS ARE TAKEN GO TO THE SIDE
        elif(corners.count("X")==2):
            board[1][2]="O"
        
        #GO IN THE MIDDLE IF TWO ADJACENT SIDES ARE TAKEN
        elif((board[0][1] == "X" and board[1][2]=="X") or (board[1][0] == "X" and board[2][1]=="X") or 
            (board[0][1] == "X" and board[1][0]=="X") or (board[2][1] == "X" and board[1][2]=="X")):
            board[1][1]="O"
            
        #THEN IF THE CORNERS ARE FREE TAKE THEM
        elif (board[2][2] != "X"):
            board[2][2] = "O"
        elif(board[2][2] == "X"):
            board[2][0] = "O"
        
        else:
            #OTHERWISE GO RANDOMLY
            empty_positions = get_empty_positions(board)
            move = random.choice(empty_positions)
            board[move[0]][move[1]] = "O"
                
    else:
        #Win if it's possible
        if(check_block(board,"O")):
            winning_move(board)
            
        #Block if it's needed
        elif(check_block(board, "X")):
            block(board)
        
        #go in the middle if it's open 
        elif(board[1][1] == " "):
            board[1][1] = "O"
        
        #OTHERWISE PUT TWO IN A ROW IF THERE ARE THREE FREE
        elif(check_attack(board, "O")):            
            attack(board)
            
        #If there's nothing just do random placement
        else:
            empty_positions = get_empty_positions(board)
            move = random.choice(empty_positions)
            board[move[0]][move[1]] = "O"
      
            
    
    

def tic_tac_toe():
    board = [[" " for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic Tac Toe!")
    user_first = input("Do you want to go first? (y/n): ").lower() == 'y'
    move=0
    for _ in range(9):
        print_board(board)
        if user_first:
            user_move(board)
            if check_winner(board, "X"):
                print_board(board)
                print("Congratulations! You win!")
                return
            user_first = False
        else:
            move+=1
            computer_move(board, move)
            if check_winner(board, "O"):
                print_board(board)
                print("Computer wins! Better luck next time.")
                return
            user_first = True
        
        if not get_empty_positions(board):
            print_board(board)
            print("It's a draw!")
            return

if __name__ == "__main__":
    tic_tac_toe()
