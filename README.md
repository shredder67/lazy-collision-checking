# Final project for the course "Planning in AI and Robotics"
### Description

Asymptotically-optimal sampling-based motion planners, like RRT*, perform vast amounts of collision checking,and are hence rather slow to converge in complex problems where collision checking is relatively expensive. In our project we investigate two variants of modifications to algorithms, named Lazy-PRM* and Lazy-RRG*, that eliminate the majority of collision checks using a lazy strategy. They are sampling based, any-time,and asymptotically complete algorithms that grow a network of feasible vertices connected by edges. Notably, collision checks for edges are deferred until a
superior path to the goal is identified. 

Our goal with project is to implement Lazy-PRM* and Lazy-RRG* and test their efficiency superiority to their predecessors (PRM* and RRG** accordingly) in terms of edge checking and complexity within selected 2D Environment (analogous to ones in mentioned paper).

### Results
...

### How to run
...

### References

1. [paper](https://ieeexplore.ieee.org/document/7139603) (for access to actual pdf refer to file in canvas)

### Authors

1. Sergey Karpukhin, Lazy RRT*, [@shredder67](https://github.com/shredder67/)
2. Maksim Osipenko, RRT*, [@MOOsipenko](https://github.com/MOOsipenko)
3. Aikun Bexultanova, PRM*, [@fokrey](https://github.com/fokrey)
4. Alisa Kalacheva, Lazy PRM*, [@AlicePH](https://github.com/AlicePH)
