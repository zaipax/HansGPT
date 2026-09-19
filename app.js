const MATRIX_SIZE = 32;
const COLUMN_COUNT = 20;
const FONT_FAMILY =
  '"Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimHei", sans-serif';

const input = document.querySelector("#character-input");
const clearButton = document.querySelector("#clear-button");
const counter = document.querySelector("#character-count");
const board = document.querySelector("#matrix-board");

const canvases = [];
const slots = [];
let previousCharacters = [];
let scheduledFrame = null;

function createSlot(index) {
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
  board.append(slot);

  canvases.push(canvas);
  slots.push(slot);
  previousCharacters.push(null);
}

function syncSlotCount(count) {
  while (slots.length < count) {
    createSlot(slots.length);
  }

  while (slots.length > count) {
    slots.pop().remove();
    canvases.pop();
    previousCharacters.pop();
  }
}

/**
 * 把输入内容排进每行 20 个的无限画布。
 * 连续文字自动换行；手动换行会在下一个字符出现时补齐当前行。
 */
function layoutText(value) {
  const characters = [];
  let characterCount = 0;
  let pendingLineBreaks = 0;

  for (const character of Array.from(value.replaceAll("\r", ""))) {
    if (character === "\n") {
      pendingLineBreaks += 1;
      continue;
    }

    if (pendingLineBreaks > 0) {
      const remainder = characters.length % COLUMN_COUNT;
      let emptySlots = 0;

      if (characters.length === 0) {
        emptySlots = pendingLineBreaks * COLUMN_COUNT;
      } else if (remainder === 0) {
        emptySlots = Math.max(0, pendingLineBreaks - 1) * COLUMN_COUNT;
      } else {
        emptySlots =
          COLUMN_COUNT - remainder + (pendingLineBreaks - 1) * COLUMN_COUNT;
      }

      characters.push(...Array(emptySlots).fill(""));
      pendingLineBreaks = 0;
    }

    characters.push(character);
    characterCount += 1;
  }

  return {
    characterCount,
    characters,
    rowCount: Math.ceil(characters.length / COLUMN_COUNT),
  };
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

function updateSlot(character, index) {
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

  previousCharacters[index] = character;
}

function render() {
  scheduledFrame = null;
  const result = layoutText(input.value);

  syncSlotCount(result.characters.length);
  result.characters.forEach(updateSlot);

  const rowLabel = result.rowCount === 0 ? "" : ` · ${result.rowCount} 行`;
  counter.textContent = `${result.characterCount} 字${rowLabel}`;
  clearButton.disabled = input.value.length === 0;
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

render();

// 系统字体加载完成后再画一次，避免字体切换造成字形尺寸不一致。
document.fonts?.ready.then(() => {
  previousCharacters.fill(null);
  render();
});
