import { NavLink } from "react-router-dom";
import { useLearningStore } from "../store/useLearningStore";

function NavItem({ to, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        [
          "rounded-full px-4 py-2 text-sm font-semibold transition",
          isActive ? "bg-ink text-white" : "text-slate-600 hover:bg-white",
        ].join(" ")
      }
    >
      {label}
    </NavLink>
  );
}

export default function AppShell({ children }) {
  const userInput = useLearningStore((state) => state.userInput);

  return (
    <div className="mx-auto min-h-screen max-w-7xl px-4 pb-10 pt-6 md:px-8">
      <header className="glass mb-6 flex flex-col gap-4 rounded-2xl p-4 shadow-card md:flex-row md:items-center md:justify-between">
        <div>
          <p className="font-display text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">PulseLearn AI</p>
          <h1 className="font-display text-2xl font-bold text-ink md:text-3xl">Adaptive Learning, Tuned To Every Click</h1>
        </div>
        <div className="flex items-center gap-2 rounded-full bg-slate-100 p-1">
          <NavItem to="/onboarding" label="Onboarding" />
          <NavItem to="/dashboard" label="Dashboard" />
        </div>
      </header>

      {userInput ? (
        <section className="mb-5 flex flex-wrap items-center gap-3">
          <span className="rounded-full bg-white/80 px-3 py-1 text-sm font-semibold text-slate-700 shadow-sm">
            Focus: {userInput.subject_weakness}
          </span>
          <span className="rounded-full bg-white/80 px-3 py-1 text-sm text-slate-700 shadow-sm">
            Quiz: {Number(userInput.quiz_score).toFixed(0)}
          </span>
          <span className="rounded-full bg-white/80 px-3 py-1 text-sm text-slate-700 shadow-sm">
            Attendance: {Number(userInput.attendance).toFixed(0)}%
          </span>
        </section>
      ) : null}

      <main>{children}</main>
    </div>
  );
}
