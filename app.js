const GRID_SIZE = 32;
const MESSAGE = "我爱你";
const FONT_FAMILY =
  '"Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimHei", sans-serif';

/**
 * 把一个汉字绘制到 32×32 的离屏画布，并返回每个像素的透明度。
 * 浏览器的字体抗锯齿会自然产生边缘亮度，让点阵笔画更柔和。
 */
function rasterizeCharacter(character) {
  const canvas = document.createElement("canvas");
  canvas.width = GRID_SIZE;
  canvas.height = GRID_SIZE;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.clearRect(0, 0, GRID_SIZE, GRID_SIZE);
  context.fillStyle = "#ffffff";
  context.font = `900 29px ${FONT_FAMILY}`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(character, GRID_SIZE / 2, GRID_SIZE / 2 + 1);

  const pixels = context.getImageData(0, 0, GRID_SIZE, GRID_SIZE).data;
  return Array.from({ length: GRID_SIZE * GRID_SIZE }, (_, index) => {
    return pixels[index * 4 + 3] / 255;
  });
}

function createGlyphCard(character, index) {
  const card = document.createElement("article");
  card.className = "glyph-card";

  const matrix = document.createElement("div");
  matrix.className = "matrix";
  matrix.setAttribute("role", "img");
  matrix.setAttribute("aria-label", `${character}字，32乘32点阵`);

  const fragment = document.createDocumentFragment();
  const pixels = rasterizeCharacter(character);

  pixels.forEach((alpha) => {
    const cell = document.createElement("span");
    cell.className = alpha > 0.08 ? "cell is-on" : "cell";
    cell.setAttribute("aria-hidden", "true");

    if (alpha > 0.08) {
      cell.style.setProperty("--intensity", Math.max(0.22, alpha).toFixed(2));
    }

    fragment.appendChild(cell);
  });

  matrix.appendChild(fragment);

  const meta = document.createElement("div");
  meta.className = "glyph-meta";
  meta.innerHTML = `
    <span class="glyph-label">${character}</span>
    <span>字符 ${index + 1} · ${GRID_SIZE * GRID_SIZE} 格</span>
  `;

  card.append(matrix, meta);
  return card;
}

function renderMessage() {
  const display = document.querySelector("#matrix-display");
  const cards = [...MESSAGE].map(createGlyphCard);
  display.replaceChildren(...cards);
}

renderMessage();

