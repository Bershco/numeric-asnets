(define (problem ${instance_name})
  (:domain fo-counters)
  (:objects
    ${counters} - counter
  )

  (:init
    (= (max_int) ${max_int})
    ${counter_values}

    ${rate_values}
    (= (total-cost) 0)
  )

  (:goal (and
    ${goal_conditions}
  ))
  (:metric minimize (total-cost))
)
