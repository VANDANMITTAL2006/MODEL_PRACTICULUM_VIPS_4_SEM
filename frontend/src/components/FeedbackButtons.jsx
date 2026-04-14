export default function FeedbackButtons({ onAction, disabled = false, compact = false }) {
  const baseClasses = compact ? "px-3 py-1.5 text-xs" : "px-3 py-2 text-sm";

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        disabled={disabled}
        className={`${baseClasses} rounded-full bg-emerald-50 font-semibold text-emerald-700 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-60`}
        onClick={() => onAction("like")}
      >
        Thumbs up
      </button>
      <button
        type="button"
        disabled={disabled}
        className={`${baseClasses} rounded-full bg-rose-50 font-semibold text-rose-700 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-60`}
        onClick={() => onAction("dislike")}
      >
        Thumbs down
      </button>
      <button
        type="button"
        disabled={disabled}
        className={`${baseClasses} rounded-full bg-slate-100 font-semibold text-slate-600 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-60`}
        onClick={() => onAction("skip")}
      >
        Skip
      </button>
    </div>
  );
}
