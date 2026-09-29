import { markdownBlocks } from "./chat_view.mjs";

export function renderMarkdown(container, source) {
  container.replaceChildren();
  for (const block of markdownBlocks(source)) {
    if (block.type === "code") {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = block.text;
      if (block.language) {
        const label = document.createElement("span");
        label.className = "code-label";
        label.textContent = block.language;
        pre.append(label);
      }
      pre.append(code);
      container.append(pre);
      continue;
    }
    if (block.type === "heading") {
      const heading = document.createElement(block.level === 1 ? "h3" : "h4");
      appendInlines(heading, block.inlines);
      container.append(heading);
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

function appendInlines(parent, tokens) {
  for (const token of tokens) {
    if (token.type === "text") {
      parent.append(document.createTextNode(token.text));
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
