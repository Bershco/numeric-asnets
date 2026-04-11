(define (problem ${instance_name})
  (:domain mt-block-grouping)
  (:objects
    ${blocks} - block
  )

  (:init
    ${initial_fluents}
    (= (max_x) ${max_coord} )
    (= (min_x) 1 )
    (= (max_y) ${max_coord} )
    (= (min_y) 1 )
  )

  (:goal (and 
    ${goal_conditions}
  ))
)
