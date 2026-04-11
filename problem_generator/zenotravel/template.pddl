(define (problem ${instance_name})
(:domain zenotravel)
(:objects
    ${aircraft} - aircraft
    ${people} - person
    ${cities} - city
)
(:init
    ${initial_statements}
)
(:goal (and
    ${goal_conditions}
))
(:metric  minimize (total-fuel-used) )
)
