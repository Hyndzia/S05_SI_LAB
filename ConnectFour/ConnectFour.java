package ConnectFour;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import sac.game.AlphaBetaPruning;
import sac.game.GameSearchAlgorithm;
import sac.game.GameSearchConfigurator;
import sac.game.GameState;
import sac.game.GameStateImpl;
import sac.game.MinMax;

public class ConnectFour extends GameStateImpl{
    public static final int m = 8;
    public static final int n = 8;

    public static final byte X = 1;
    public static final byte O = -1;
    public static final byte E = 0;

    private static final String[] SYMBOLS = {"O", ".", "X"};

    public static final boolean IS_AI_MAX = false;
    public static final boolean IS_AI_MIN = true;

    public static final boolean CEALING_RULE_ON = true;

    public byte[][] board;

    public ConnectFour() {
        board = new byte[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                board[i][j] = E;
    }

    public ConnectFour(ConnectFour other) {
        board = new byte[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                board[i][j] = other.board[i][j];
        setMaximizingTurnNow(other.isMaximizingTurnNow());
    }

    public byte[][] getBoard() {
        return board;
    }
    public int getRows() {
        return m;
    }
    public int getCols() {
        return n;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < m; i++) {
            sb.append("|");
            for (int j = 0; j < n; j++) {
                sb.append(SYMBOLS[board[i][j] + 1]);
                sb.append("|");
            }
            sb.append("\n");
        }
        sb.append(" ");
        for (int j = 0; j < n; j++) {
            sb.append(j);
            sb.append(" ");
        }
        return sb.toString();
    }

    public boolean move(int column) {
        for (int i = m - 1; i >= 0; i--)
            if(board[i][column] == E) {
                board[i][column] = (isMaximizingTurnNow()) ? X : O;
                setMaximizingTurnNow(!isMaximizingTurnNow());
                return true;
            }
        return false;
    }

    public boolean checkWin() {
        int[] di = new int[] {0, -1, -1, -1};
        int[] dj = new int[] {+1, +1, 0, -1};

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                for(int k = 0; k < 4; k++) {
                    byte sum = 0;
                    for(int q = 0; q < 4; q++) {
                        int index_i = i + q * di[k];
                        int index_j = j + q * dj[k];
                        if ((index_i >= 0) && (index_j >= 0) && (index_j < n)) {
                            sum += board[index_i][index_j];
                        }
                    }
                    if (Math.abs(sum) == 4)
                        return true;
                }
            }
        }
        if (CEALING_RULE_ON)
            for (int j = 0; j < n; j++)
                if(board[0][j] != E)
                    return true;
        return false;
    }

    @Override
    public int hashCode() {
        byte[] flat = new byte[m * n];
        int k = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                flat[k++] = board[i][j];
        return Arrays.hashCode(flat);
    }

    @Override
    public List<GameState> generateChildren() {
        List<GameState> children = new ArrayList<GameState>();
        for (int j = 0; j < n; j++) {
            ConnectFour child = new ConnectFour(this);
            child.setMoveName(Integer.toString(j));
            if (child.move(j)) {
                children.add(child);
            }
        }
        return children;
    }

    public static void main(String args[]) {
        ConnectFour c4 = new ConnectFour();
        Scanner scanner = new Scanner(System.in);
        boolean moveOk;
        ConnectFour.setHFunction(new ConnectFourHeuristic());
        GameSearchConfigurator config = new GameSearchConfigurator();
        config.setDepthLimit(4.5);
        GameSearchAlgorithm algo = new AlphaBetaPruning();
        algo.setConfigurator(config);

        while(true) {
            System.out.println(c4);
            if (c4.checkWin())
            {
                System.out.println("O's WIN!");
                break;
            }
            if (IS_AI_MAX) {
                algo.setInitial(c4);
                algo.execute();
                System.out.println("AI time [ms]: " + algo.getDurationTime());
                System.out.println("AI states analyzed: " + algo.getClosedStatesCount());
                System.out.println("AI move scores: " + algo.getMovesScores());
                String bestMoveName = algo.getFirstBestMove();
                System.out.println("AI best move:" + bestMoveName);
                c4.move(Integer.valueOf(bestMoveName));
            }
            else {
                do {
                    System.out.println("X's turn, make your move: ");
                    String moveAsString = scanner.nextLine(); //scanner.nextInt()
                    int column = Integer.valueOf(moveAsString);
                    moveOk = c4.move(column);
                } while (!moveOk);
            }
            System.out.println(c4);
            if (c4.checkWin()) {
                System.out.println("X's WIN!");
                break;
            }
            if (IS_AI_MIN) {
                algo.setInitial(c4);
                algo.execute();
                System.out.println("AI time [ms]: " + algo.getDurationTime());
                System.out.println("AI states analyzed: " + algo.getClosedStatesCount());
                System.out.println("AI move scores: " + algo.getMovesScores());
                String bestMoveName = algo.getFirstBestMove();
                System.out.println("AI best move:" + bestMoveName);
                c4.move(Integer.valueOf(bestMoveName));
            }
            else {
                do {
                    System.out.println("O's turn, make your move: ");
                    String moveAsString = scanner.nextLine(); //scanner.nextInt()
                    int column = Integer.valueOf(moveAsString);
                    moveOk = c4.move(column);
                } while (!moveOk);
            }
        }
    }
}