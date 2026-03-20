import { NavLink } from "react-router-dom";
import { Database, MapPin } from "lucide-react";
import { GeoAiMark } from "@/components/GeoAiMark";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
    isActive
      ? "bg-primary/20 text-primary"
      : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
  }`;

const SuiteNav = () => (
  <nav className="flex shrink-0 items-center gap-1 border-b border-border/70 bg-card/90 px-3 py-1.5 backdrop-blur-md">
    <NavLink to="/" end className={linkClass}>
      <GeoAiMark className="h-3.5 w-3.5 shrink-0" />
      CV-Scan
    </NavLink>
    <NavLink to="/venues" className={linkClass}>
      <MapPin className="h-3.5 w-3.5" />
      Venues
    </NavLink>
    <NavLink to="/spatial" className={linkClass}>
      <Database className="h-3.5 w-3.5" />
      Spatial Visualizer
    </NavLink>
  </nav>
);

export default SuiteNav;
