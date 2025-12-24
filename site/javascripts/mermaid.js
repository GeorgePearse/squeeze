(() => {
  const renderMermaid = () => {
    if (!window.mermaid) return;

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme:
        document.body?.getAttribute("data-md-color-scheme") === "slate"
          ? "dark"
          : "default",
    });

    const nodes = document.querySelectorAll(".mermaid");
    nodes.forEach((node) => node.removeAttribute("data-processed"));

    if (typeof window.mermaid.run === "function") {
      window.mermaid.run({ nodes });
      return;
    }

    if (typeof window.mermaid.init === "function") {
      window.mermaid.init(undefined, nodes);
    }
  };

  if (window.document$?.subscribe) {
    window.document$.subscribe(renderMermaid);
  } else {
    document.addEventListener("DOMContentLoaded", renderMermaid);
  }
})();

