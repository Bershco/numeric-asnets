(define (problem ${instance_name})
  (:domain delivery)
  (:objects
    ${rooms_list} - room
    ${items_list} - item
    ${bots_list} - bot
    ${arms_list} - arm
  )

  (:init
    ${initial_fluents}

    ${initial_predicates}
  )

  (:goal (and
    ${goal_conditions}
  ))

  (:metric minimize (cost))
)
