package ConnectFour;

import sac.State;
import sac.StateFunction;

        public class ConnectFourHeuristic extends StateFunction {

            @Override
            public double calculate(State state) {
                ConnectFour c4 = (ConnectFour) state;

                // Sprawdzenie, czy gra została już wygrana
                boolean isWon = c4.checkWin();
                if (isWon) {
                    return (c4.isMaximizingTurnNow()) ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
                }

                int p1_result = calculateResult(c4, 1);
                int p2_result = calculateResult(c4, 2);


                return p1_result - p2_result;
            }

            private int calculateResult(ConnectFour c4, int player) {
                byte[][] board = c4.getBoard(); //Pobranie planszy
                int result = 0;

                //Iteracja po planszy i ocena sumy długości podciągów w różnych kierunkach
                for (int row = 0; row < c4.getRows(); row++) {
                    for (int col = 0; col < c4.getCols(); col++) {
                        if (board[row][col] == player) {
                            result += checkDirection(board, row, col, 1, 0, player); // poziomo
                            result += checkDirection(board, row, col, 0, 1, player); // pionowo
                            result += checkDirection(board, row, col, 1, 1, player); //ukos w prawo
                            result += checkDirection(board, row, col, 1, -1, player); //ukos w lewo
                        }
                    }
                }
                return result;
            }

            private int checkDirection(byte[][] board, int row, int col, int rowDelta, int colDelta, int player) {
                int length = 0;
                while (row >= 0 && row < board.length && col >= 0 && col < board[0].length &&
                        board[row][col] == player) {
                    length++;
                    row += rowDelta;
                    col += colDelta;
                }
                return length;
            }
        }





