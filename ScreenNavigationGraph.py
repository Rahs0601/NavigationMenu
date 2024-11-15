from collections import defaultdict, deque
from typing import List, Dict, Set, Optional

class ScreenNavigationGraph:
    def __init__(self):
        """Initialize an empty graph for screen navigation."""
        self.graph = defaultdict(set)
        self.screens = set()
    
    def add_screen(self, screen: str) -> None:
        """Add a screen to the graph."""
        self.screens.add(screen)
    
    def add_navigation(self, from_screen: str, to_screen: str) -> None:
        """Add a navigation path between two screens."""
        self.screens.add(from_screen)
        self.screens.add(to_screen)
        self.graph[from_screen].add(to_screen)
        # Adding bi-directional navigation - comment out if one-way navigation is needed
        self.graph[to_screen].add(from_screen)
    
    def get_navigation_path(self, start_screen: str, target_screen: str) -> Optional[List[str]]:
        """
        Find the shortest path between start_screen and target_screen.
        Returns None if no path exists.
        """
        if start_screen not in self.screens or target_screen not in self.screens:
            print("Invalid start or target screen.")
            return None
        
        # Using BFS to find shortest path
        queue = deque([[start_screen]])
        visited = {start_screen}
        
        while queue:
            path = queue.popleft()
            current_screen = path[-1]
            
            if current_screen == target_screen:
                return path
            
            for next_screen in self.graph[current_screen]:
                if next_screen not in visited:
                    visited.add(next_screen)
                    queue.append(path + [next_screen])
                    # queue.append(path + [screenNames[next_screen]])
        
        return None

    def get_all_screens(self) -> Set[str]:
        """Return all screens in the graph."""
        return self.screens

    def get_connected_screens(self, screen: str) -> Set[str]:
        """Return all screens directly connected to the given screen."""
        return self.graph[screen]