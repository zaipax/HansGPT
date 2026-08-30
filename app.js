const MATRIX_SIZE = 32;
const COLUMN_COUNT = 10;
const ROW_COUNT = 18;
const SLOT_COUNT = COLUMN_COUNT * ROW_COUNT;
const FONT_FAMILY =
  '"Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimHei", sans-serif';

const input = document.querySelector("#character-input");
const clearButton = document.querySelector("#clear-button");
const counter = document.querySelector("#character-count");
const board = document.querySelector("#matrix-board");

const canvases = [];
const slots = [];
let previousCharacters = Array(SLOT_COUNT).fill(null);
let scheduledFrame = null;

function createBoard() {
  const fragment = document.createDocumentFragment();

  for (let index = 0; index < SLOT_COUNT; index += 1) {
    const slot = document.createElement("div");
    const screen = document.createElement("div");
    const canvas = document.createElement("canvas");
    const number = document.createElement("span");

    slot.className = "matrix-slot";
    slot.setAttribute("aria-hidden", "true");

    screen.className = "matrix-screen";
    canvas.width = MATRIX_SIZE;
    canvas.height = MATRIX_SIZE;
    canvas.setAttribute("aria-hidden", "true");

    number.className = "slot-number";
    number.textContent = String(index + 1).padStart(3, "0");
    number.setAttribute("aria-hidden", "true");

    screen.append(canvas);
    slot.append(screen, number);
    fragment.append(slot);

    canvases.push(canvas);
    slots.push(slot);
  }

  board.append(fragment);
}

/**
 * 把输入内容排进固定的 10×18 字位。
 * 普通字符顺序填充；换行符会把光标移到下一行开头。
 */
function layoutText(value) {
  const characters = Array(SLOT_COUNT).fill("");
  let acceptedValue = "";
  let position = 0;
  let characterCount = 0;

  for (const character of Array.from(value.replaceAll("\r", ""))) {
    if (character === "\n") {
      const nextRow =
        position % COLUMN_COUNT === 0
          ? position + COLUMN_COUNT
          : position + COLUMN_COUNT - (position % COLUMN_COUNT);

      if (nextRow >= SLOT_COUNT) break;

      position = nextRow;
      acceptedValue += character;
      continue;
    }

    if (position >= SLOT_COUNT) break;

    characters[position] = character;
    acceptedValue += character;
    position += 1;
    characterCount += 1;
  }

  return { acceptedValue, characterCount, characters };
}

function drawCharacter(canvas, character) {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, MATRIX_SIZE, MATRIX_SIZE);

  if (!character || character.trim() === "") return;

  context.fillStyle = "#ff4f7b";
  context.font = `900 29px ${FONT_FAMILY}`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(character, MATRIX_SIZE / 2, MATRIX_SIZE / 2 + 1);
}

function render() {
  scheduledFrame = null;
  const result = layoutText(input.value);

  if (input.value !== result.acceptedValue) {
    input.value = result.acceptedValue;
  }

  result.characters.forEach((character, index) => {
    if (character === previousCharacters[index]) return;

    drawCharacter(canvases[index], character);
    slots[index].classList.toggle("is-filled", character !== "");

    if (character !== "") {
      const row = Math.floor(index / COLUMN_COUNT) + 1;
      const column = (index % COLUMN_COUNT) + 1;
      slots[index].setAttribute("role", "img");
      slots[index].setAttribute(
        "aria-label",
        `第 ${row} 行第 ${column} 列：${character}，32乘32点阵`,
      );
      slots[index].removeAttribute("aria-hidden");
    } else {
      slots[index].removeAttribute("role");
      slots[index].removeAttribute("aria-label");
      slots[index].setAttribute("aria-hidden", "true");
    }
  });

  previousCharacters = result.characters;
  counter.textContent = `${result.characterCount} / ${SLOT_COUNT} 字`;
  clearButton.disabled = result.acceptedValue.length === 0;
}

function scheduleRender() {
  if (scheduledFrame !== null) return;
  scheduledFrame = window.requestAnimationFrame(render);
}

input.addEventListener("input", scheduleRender);

clearButton.addEventListener("click", () => {
  input.value = "";
  render();
  input.focus();
});

createBoard();
render();

// 系统字体加载完成后再画一次，避免 Web 字体切换造成字形尺寸不一致。
document.fonts?.ready.then(() => {
  previousCharacters = Array(SLOT_COUNT).fill(null);
  render();
});
