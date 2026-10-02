import { markdownBlocks } from "./chat_view.mjs?v=3";

export function renderMarkdown(container, source) {
  container.replaceChildren();
  for (const block of markdownBlocks(source)) {
    if (block.type === "code") {
      container.append(renderCodeBlock(block));
      continue;
    }
    if (block.type === "heading") {
      const heading = document.createElement(block.level === 1 ? "h3" : "h4");
      appendInlines(heading, block.inlines);
      container.append(heading);
      continue;
    }
    if (block.type === "quote") {
      const quote = document.createElement("blockquote");
      appendInlines(quote, block.inlines);
      container.append(quote);
      continue;
    }
    if (block.type === "rule") {
      container.append(document.createElement("hr"));
      continue;
    }
    if (block.type === "list") {
      const list = document.createElement(block.ordered ? "ol" : "ul");
      for (const item of block.items) {
        const li = document.createElement("li");
        appendInlines(li, item);
        list.append(li);
      }
      container.append(list);
      continue;
    }
    const paragraph = document.createElement("p");
    appendInlines(paragraph, block.inlines);
    container.append(paragraph);
  }
}

function renderCodeBlock(block) {
  const wrap = document.createElement("div");
  wrap.className = "code-block";
  const bar = document.createElement("div");
  bar.className = "code-block-bar";
  const label = document.createElement("span");
  label.textContent = block.language || "code";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "code-copy";
  copy.textContent = "Copy";
  copy.addEventListener("click", () => {
    void copyCode(copy, block.text);
  });
  bar.append(label, copy);
  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.textContent = block.text;
  pre.append(code);
  wrap.append(bar, pre);
  return wrap;
}

async function copyCode(button, text) {
  let copied = false;
  try {
    if (navigator.clipboard && window.isSecureContext && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      copied = true;
    }
  } catch (error) {
    copied = false;
  }
  if (!copied) {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.top = "0";
    area.style.left = "0";
    area.style.opacity = "0";
    document.body.append(area);
    area.focus();
    area.select();
    copied = document.execCommand("copy");
    area.remove();
  }
  button.textContent = copied ? "Copied" : "Copy";
  window.setTimeout(() => {
    if (button.isConnected) {
      button.textContent = "Copy";
    }
  }, 1500);
}

function appendInlines(parent, tokens) {
  for (const token of tokens) {
    if (token.type === "text") {
      parent.append(document.createTextNode(token.text));
    } else if (token.type === "break") {
      parent.append(document.createElement("br"));
    } else if (token.type === "code") {
      const code = document.createElement("code");
      code.textContent = token.text;
      parent.append(code);
    } else if (token.type === "strong") {
      const strong = document.createElement("strong");
      strong.textContent = token.text;
      parent.append(strong);
    } else if (token.type === "em") {
      const em = document.createElement("em");
      em.textContent = token.text;
      parent.append(em);
    } else if (token.type === "link") {
      const link = document.createElement("a");
      link.href = token.href;
      link.rel = "noopener noreferrer";
      link.target = "_blank";
      link.textContent = token.text;
      parent.append(link);
    }
  }
}
