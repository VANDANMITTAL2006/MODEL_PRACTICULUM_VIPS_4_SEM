import { useEffect } from "react";
import { useLearningStore } from "../store/useLearningStore";

export function useSessionTracker() {
  const pushAnalytics = useLearningStore((state) => state.pushAnalytics);

  useEffect(() => {
    pushAnalytics({ event: "session_started" });

    return () => {
      pushAnalytics({ event: "session_ended" });
    };
  }, [pushAnalytics]);
}
