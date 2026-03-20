import { GeoAiMark } from "@/components/GeoAiMark";

const Navbar = () => {
  return (
    <nav className="sticky top-0 z-40 border-b border-border/50 bg-background/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GeoAiMark className="h-9 w-9 shrink-0" />
          <span className="font-display font-bold text-foreground tracking-tight">
            Geo<span className="text-primary">AI</span>
          </span>
          <span className="text-xs font-mono text-muted-foreground ml-2 hidden sm:inline">
            v2.4.1
          </span>
        </div>

        <div className="flex items-center gap-6">
          <span className="text-xs font-mono text-muted-foreground hidden md:inline">
            ENTRANCE DETECTION SYSTEM
          </span>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-geo-success animate-pulse" />
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
