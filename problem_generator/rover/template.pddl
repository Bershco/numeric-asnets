(define (problem ${instance_name}) (:domain rover)
(:objects
    general - lander
    colour high_res low_res - mode
    ${rovers} - rover
    ${stores} - store
    ${waypoints} - waypoint
    ${cameras} - camera
    ${objectives} - objective
)
(:init
    ${initial_predicates}
    ${initial_fluents}
)

(:goal (and
    ${goal_conditions}
))

(:metric minimize (recharges))
)
