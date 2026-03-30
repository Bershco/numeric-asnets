;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem ${instance_name})
  (:domain ${domain_name})
  (:objects
    ${counters_list} - counter
  )

  (:init
    (= (max_int) ${max_int_value})
    ${counters_initial_values}
  )

  (:goal (and
    ${counters_final_values}
  ))


)

