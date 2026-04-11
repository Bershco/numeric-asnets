(define (problem ${instance_name})
(:domain TPP-Metric)
(:objects
    ${markets} - market
    depot0 - depot
    truck0 - truck
    ${goods} - goods)
(:init
    ${initial_statements}
)

(:goal (and
    ${goal_conditions}
))

(:metric minimize (total-cost))
)
