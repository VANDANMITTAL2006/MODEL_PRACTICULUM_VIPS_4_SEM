import { useEffect, useMemo, useState } from "react";

function formatTime(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export default function LearningTracker({ itemTitle, onComplete, onTick, disabled = false }) {
  const [elapsed, setElapsed] = useState(0);
  const [completed, setCompleted] = useState(false);

  useEffect(() => {
    setElapsed(0);
    setCompleted(false);
  }, [itemTitle]);

  useEffect(() => {
    if (disabled || completed) {
      return undefined;
    }

    const interval = window.setInterval(() => {
      setElapsed((value) => {
        const next = value + 1;
        if (next % 5 === 0) {
          onTick(next);
        }
        return next;
      });
    }, 1000);

    return () => window.clearInterval(interval);
  }, [completed, disabled, onTick]);

  const milestone = useMemo(() => {
    if (elapsed < 30) {
      return "Warm-up phase: skim the key concept.";
    }
    if (elapsed < 90) {
      return "Deep focus: connect examples with prior lessons.";
    }
    return "Mastery mode: summarize this concept in your own words.";
  }, [elapsed]);

  const completeLesson = () => {
    setCompleted(true);
    onComplete(elapsed);
  };

  return (
    <aside className="glass rounded-2xl p-5 shadow-card">
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Learning Tracker</p>
      <p className="mb-3 font-display text-3xl font-bold text-ink">{formatTime(elapsed)}</p>
      <p className="mb-5 text-sm text-slate-600">{milestone}</p>
      <button
        type="button"
        disabled={completed}
        className="w-full rounded-xl bg-mint px-4 py-2 text-sm font-semibold text-white transition hover:brightness-95 disabled:cursor-not-allowed disabled:bg-slate-300"
        onClick={completeLesson}
      >
        {completed ? "Completed" : "Mark as complete"}
      </button>
    </aside>
  );
}
