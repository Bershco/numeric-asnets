;;Instance with ${x_size}x${y_size}x${z_size} points
(define (problem ${instance_name}) (:domain drone)
(:objects 
${locations} - location
) 
(:init (= (x) 0) (= (y) 0) (= (z) 0)
 (= (min_x) 0)  (= (max_x) ${x_size}) 
 (= (min_y) 0)  (= (max_y) ${y_size}) 
 (= (min_z) 0)  (= (max_z) ${z_size}) 
${coordinate_fluents}
(= (battery-level) ${battery_level})
(= (battery-level-full) ${battery_level})
)
(:goal (and 
${goal_conditions}
(= (x) 0) (= (y) 0) (= (z) 0) ))
);; end of the problem instance
