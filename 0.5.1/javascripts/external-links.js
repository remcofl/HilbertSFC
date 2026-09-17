function markExternalLinks() {
  document.querySelectorAll(".md-content .md-typeset a").forEach((link) => {
    if (link.classList.contains("autorefs") || link.classList.contains("headerlink")) {
      return;
    }

    try {
      const destination = new URL(link.href, document.baseURI);
      if (destination.origin !== window.location.origin) {
        link.classList.add("external-link");
      }
    } catch {
      // Leave malformed or non-web links unmodified.
    }
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(markExternalLinks);
} else {
  document.addEventListener("DOMContentLoaded", markExternalLinks);
}
