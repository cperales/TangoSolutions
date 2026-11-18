const BOARD_SIZE = 6;
let solutions = [];
let currentSolution = null;
let board = [];

fetch('solutions.json')
  .then(r => r.json())
  .then(data => {
    solutions = data;
    resetGame();
  });

function emptyBoard() {
  return Array.from({length: BOARD_SIZE}, () => Array(BOARD_SIZE).fill(null));
}

function resetGame() {
  currentSolution = solutions[Math.floor(Math.random() * solutions.length)];
  board = emptyBoard();
  renderBoard();
  document.getElementById('status').textContent = '';
}

function renderBoard() {
  const container = document.getElementById('board');
  container.innerHTML = '';
  for (let i = 0; i < BOARD_SIZE; i++) {
    const rowDiv = document.createElement('div');
    rowDiv.className = 'row';
    for (let j = 0; j < BOARD_SIZE; j++) {
      const cellDiv = document.createElement('div');
      cellDiv.className = 'cell';
      const val = board[i][j];
      cellDiv.textContent = val === null ? '' : val;
      cellDiv.onclick = () => {
        let v = board[i][j];
        if (v === null) v = 0;
        else if (v === 0) v = 1;
        else v = null;
        board[i][j] = v;
        renderBoard();
      };
      rowDiv.appendChild(cellDiv);
    }
    container.appendChild(rowDiv);
  }
}

function checkSolution() {
  for (let i = 0; i < BOARD_SIZE; i++) {
    for (let j = 0; j < BOARD_SIZE; j++) {
      if (board[i][j] !== currentSolution[i][j]) {
        document.getElementById('status').textContent = 'Incorrect.';
        return;
      }
    }
  }
  document.getElementById('status').textContent = 'You solved it!';
}

document.getElementById('check').onclick = checkSolution;
document.getElementById('reset').onclick = resetGame;
