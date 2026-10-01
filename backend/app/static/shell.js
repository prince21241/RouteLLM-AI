const ROUTES = {
  "/": "chat",
  "/overview": "overview",
  "/dashboard": "overview",
  "/requests": "requests",
  "/costs": "costs",
  "/providers": "providers",
};

const TITLES = {
  chat: "Chat",
  overview: "Overview",
  requests: "Requests",
  costs: "Costs & Usage",
  providers: "Providers & Models",
};

const navToggle = document.querySelector("#nav-toggle");
const backdrop = document.querySelector("#nav-backdrop");

function currentRoute() {
  if (location.hash.startsWith("#request=")) {
    return "requests";
  }
  return ROUTES[location.pathname] || "chat";
}

function applyRoute() {
  const route = currentRoute();
  document.body.dataset.route = route;
  const title = TITLES[route] || "RouteLLM";
  const heading = document.querySelector("#page-title");
  if (heading) {
    heading.textContent = title;
  }
  document.title = `RouteLLM · ${title}`;
  document.querySelectorAll("a[data-nav]").forEach((link) => {
    if (link.dataset.nav === route) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
  closeNav();
}

function closeNav() {
  document.body.classList.remove("nav-open");
  if (navToggle) {
    navToggle.setAttribute("aria-expanded", "false");
  }
  if (backdrop) {
    backdrop.hidden = true;
  }
}

function openNav() {
  document.body.classList.add("nav-open");
  if (navToggle) {
    navToggle.setAttribute("aria-expanded", "true");
  }
  if (backdrop) {
    backdrop.hidden = false;
  }
}

if (navToggle) {
  navToggle.addEventListener("click", () => {
    if (document.body.classList.contains("nav-open")) {
      closeNav();
    } else {
      openNav();
    }
  });
}

if (backdrop) {
  backdrop.addEventListener("click", closeNav);
}

document.addEventListener("click", (event) => {
  const link = event.target.closest("a[data-nav]");
  if (!link || event.defaultPrevented || event.button !== 0) {
    return;
  }
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
    return;
  }
  const url = new URL(link.href, location.origin);
  if (url.origin !== location.origin) {
    return;
  }
  event.preventDefault();
  history.pushState(null, "", `${url.pathname}${url.search}${url.hash}`);
  applyRoute();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("nav-open")) {
    closeNav();
    navToggle?.focus();
  }
});

window.addEventListener("popstate", applyRoute);
window.addEventListener("hashchange", applyRoute);
window.routellmApplyRoute = applyRoute;
applyRoute();
