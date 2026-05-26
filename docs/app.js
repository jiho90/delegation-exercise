const app = document.querySelector("#app");

let tasks = [];
let answers = [];
let currentTask = 0;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function optionLabel(task, wnid) {
  return task.options.find((option) => option.wnid === wnid)?.label || wnid;
}

function renderIntro() {
  app.className = "shell intro";
  app.innerHTML = `
    <section class="intro-panel">
      <p class="eyebrow">Delegation condition</p>
      <h1>Image classification task</h1>
      <p>
        Classify ${tasks.length} fixed Tiny ImageNet images of dogs and cats.
        Each task shows five same-species classes and five examples per class.
        You may answer yourself or delegate the task to GoogLeNet Inception v3.
      </p>
      <button class="primary-button" type="button" data-action="start">Start</button>
    </section>
  `;
}

function renderTask() {
  const task = tasks[currentTask];
  app.className = "shell task-shell";
  app.innerHTML = `
    <div class="loading-banner" role="status" aria-live="polite">Loading next task...</div>
    <header class="task-header">
      <div>
        <p class="eyebrow">Task ${task.index + 1} of ${tasks.length}</p>
        <h1>Select the class for this image</h1>
      </div>
      <div class="progress" aria-label="Progress">
        <span style="width: ${Math.floor(((task.index + 1) / tasks.length) * 100)}%"></span>
      </div>
    </header>
    <section class="workspace">
      <figure class="focal">
        <img src="${task.image}" alt="Image to classify" decoding="async" fetchpriority="high">
      </figure>
      <div class="choices" aria-label="Class options">
        ${task.options.map((option) => `
          <form class="choice-card" data-choice="${escapeHtml(option.wnid)}">
            <button type="submit" aria-label="Choose ${escapeHtml(option.label)}">
              <span class="choice-title">${escapeHtml(option.label)}</span>
              <span class="examples">
                ${option.examples.map((example) => `<img src="${example}" alt="" loading="lazy" decoding="async">`).join("")}
              </span>
            </button>
          </form>
        `).join("")}
        <form class="delegate-card" data-delegate="true">
          <button type="submit">Delegate to AI</button>
        </form>
      </div>
    </section>
  `;
}

function setSubmitting(message) {
  const banner = document.querySelector(".loading-banner");
  if (banner) {
    banner.textContent = message;
  }
  document.body.classList.add("is-submitting");
  for (const button of document.querySelectorAll("button")) {
    button.disabled = true;
  }
}

function answerTask(answer) {
  answers[currentTask] = answer;
  currentTask += 1;
  window.setTimeout(() => {
    document.body.classList.remove("is-submitting");
    if (currentTask >= tasks.length) {
      renderResults();
    } else {
      renderTask();
    }
  }, 80);
}

function renderResults() {
  const rows = tasks.map((task, index) => {
    const answer = answers[index];
    const finalChoice = answer.choice;
    const correct = finalChoice === task.truth;
    const aiCorrect = task.ai.choice === task.truth;
    return { task, answer, finalChoice, correct, aiCorrect };
  });
  const correctCount = rows.filter((row) => row.correct).length;
  const delegatedCount = rows.filter((row) => row.answer.mode === "delegated").length;
  const aiCorrectCount = rows.filter((row) => row.aiCorrect).length;

  app.className = "shell results-shell";
  app.innerHTML = `
    <header class="results-header">
      <div>
        <p class="eyebrow">Results</p>
        <h1>${correctCount} / ${tasks.length} correct</h1>
      </div>
      <div class="metric-grid">
        <div class="metric">
          <span>${((correctCount / tasks.length) * 100).toFixed(1)}%</span>
          <p>Final accuracy</p>
        </div>
        <div class="metric">
          <span>${delegatedCount}</span>
          <p>Delegated</p>
        </div>
        <div class="metric">
          <span>${aiCorrectCount} / ${tasks.length}</span>
          <p>AI alone</p>
        </div>
      </div>
    </header>
    <section class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Image</th>
            <th>Final answer</th>
            <th>Correct class</th>
            <th>AI would have picked</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td><img class="thumb" src="${row.task.image}" alt="" loading="lazy" decoding="async"></td>
              <td>
                <strong>${escapeHtml(optionLabel(row.task, row.finalChoice))}</strong>
                <span>${row.answer.mode === "delegated" ? "Delegated to AI" : "Answered by subject"}</span>
              </td>
              <td>${escapeHtml(optionLabel(row.task, row.task.truth))}</td>
              <td>
                <strong>${escapeHtml(optionLabel(row.task, row.task.ai.choice))}</strong>
                <span>${(row.task.ai.confidence * 100).toFixed(2)}% among options</span>
                <span>Top ImageNet label: ${escapeHtml(row.task.ai.topImagenetLabel)}</span>
              </td>
              <td><span class="status ${row.correct ? "ok" : "bad"}">${row.correct ? "Correct" : "Incorrect"}</span></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </section>
    <button class="secondary-button" type="button" data-action="restart">Run again</button>
  `;
}

app.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) {
    return;
  }
  const action = button.dataset.action;
  if (action === "start" || action === "restart") {
    answers = [];
    currentTask = 0;
    renderTask();
  }
});

app.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.target;
  const task = tasks[currentTask];
  if (form.dataset.delegate === "true") {
    setSubmitting("Delegating to AI...");
    answerTask({ mode: "delegated", choice: task.ai.choice });
    return;
  }
  const choice = form.dataset.choice;
  if (choice) {
    setSubmitting("Loading next task...");
    answerTask({ mode: "human", choice });
  }
});

fetch("tasks.json")
  .then((response) => response.json())
  .then((data) => {
    tasks = data.tasks;
    renderIntro();
  })
  .catch((error) => {
    app.className = "shell intro";
    app.innerHTML = `
      <section class="intro-panel error-panel">
        <p class="eyebrow">Load error</p>
        <h1>Could not load tasks</h1>
        <p>${escapeHtml(error.message)}</p>
      </section>
    `;
  });
