window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"], ["$$", "$$"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
  
  if (typeof document$ !== "undefined") {
    document$.subscribe(() => {
      if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetClear();
        MathJax.texReset();
        MathJax.typesetPromise();
      }
    });
  }