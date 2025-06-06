(define (problem market1)
(:domain Trader)
(:objects
            Lisbon Berlin - market
        camel0 - camel
        Food ExpensiveRugs Coffee Cattle Water Cars GummyBears Computers LaminateFloor Copper Footballs Kittens Minerals Gold Platinum DVDs TuringMachines - goods)
(:init

        (= (price Food Lisbon)    7.6)
        (= (on-sale Food Lisbon)  6)
        (= (price ExpensiveRugs Lisbon)    8.0)
        (= (on-sale ExpensiveRugs Lisbon)  10)
        (= (price Coffee Lisbon)    26.0)
        (= (on-sale Coffee Lisbon)  2)
        (= (price Cattle Lisbon)    16.0)
        (= (on-sale Cattle Lisbon)  0)
        (= (price Water Lisbon)    33.2)
        (= (on-sale Water Lisbon)  0)
        (= (price Cars Lisbon)    78.3)
        (= (on-sale Cars Lisbon)  54)
        (= (price GummyBears Lisbon)    94.0)
        (= (on-sale GummyBears Lisbon)  8)
        (= (price Computers Lisbon)    61.6)
        (= (on-sale Computers Lisbon)  56)
        (= (price LaminateFloor Lisbon)    46.8)
        (= (on-sale LaminateFloor Lisbon)  40)
        (= (price Copper Lisbon)    31.2)
        (= (on-sale Copper Lisbon)  17)
        (= (price Footballs Lisbon)    49.6)
        (= (on-sale Footballs Lisbon)  29)
        (= (price Kittens Lisbon)    70.3)
        (= (on-sale Kittens Lisbon)  0)
        (= (price Minerals Lisbon)    12.8)
        (= (on-sale Minerals Lisbon)  53)
        (= (price Gold Lisbon)    38.8)
        (= (on-sale Gold Lisbon)  2)
        (= (price Platinum Lisbon)    68.3)
        (= (on-sale Platinum Lisbon)  55)
        (= (price DVDs Lisbon)    18.0)
        (= (on-sale DVDs Lisbon)  0)
        (= (price TuringMachines Lisbon)    21.2)
        (= (on-sale TuringMachines Lisbon)  0)

        (= (price Food Berlin)    3.6)
        (= (on-sale Food Berlin)  16)
        (= (price ExpensiveRugs Berlin)    6.0)
        (= (on-sale ExpensiveRugs Berlin)  15)
        (= (price Coffee Berlin)    20.0)
        (= (on-sale Coffee Berlin)  17)
        (= (price Cattle Berlin)    6.0)
        (= (on-sale Cattle Berlin)  0)
        (= (price Water Berlin)    23.2)
        (= (on-sale Water Berlin)  20)
        (= (price Cars Berlin)    94.3)
        (= (on-sale Cars Berlin)  14)
        (= (price GummyBears Berlin)    49.6)
        (= (on-sale GummyBears Berlin)  55)
        (= (price Computers Berlin)    89.6)
        (= (on-sale Computers Berlin)  0)
        (= (price LaminateFloor Berlin)    58.8)
        (= (on-sale LaminateFloor Berlin)  10)
        (= (price Copper Berlin)    33.2)
        (= (on-sale Copper Berlin)  12)
        (= (price Footballs Berlin)    75.6)
        (= (on-sale Footballs Berlin)  0)
        (= (price Kittens Berlin)    52.3)
        (= (on-sale Kittens Berlin)  9)
        (= (price Minerals Berlin)    10.8)
        (= (on-sale Minerals Berlin)  58)
        (= (price Gold Berlin)    36.8)
        (= (on-sale Gold Berlin)  7)
        (= (price Platinum Berlin)    64.3)
        (= (on-sale Platinum Berlin)  1)
        (= (price DVDs Berlin)    16.0)
        (= (on-sale DVDs Berlin)  0)
        (= (price TuringMachines Berlin)    51.2)
        (= (on-sale TuringMachines Berlin)  0)
        (= (bought Food ) 0)
        (= (bought ExpensiveRugs ) 0)
        (= (bought Coffee) 0)
        (= (bought Cattle ) 0)
        (= (bought Water ) 0)
        (= (bought Cars ) 0)
        (= (bought GummyBears ) 0)
        (= (bought Computers ) 0)
        (= (bought LaminateFloor ) 0)
        (= (bought Copper ) 0)
        (= (bought Footballs ) 0)
        (= (bought Kittens ) 0)
        (= (bought Minerals ) 0)
        (= (bought Gold ) 0)
        (= (bought Platinum ) 0)
        (= (bought DVDs ) 0)
        (= (bought TuringMachines ) 0)
        (= (drive-cost Lisbon Berlin ) 4.3)
        (= (drive-cost Berlin Lisbon ) 4.3)
        (can-drive Lisbon Berlin)
        (can-drive Berlin Lisbon)
        (at camel0       Berlin)
        (= (cash) 100)
        (= (capacity) 20)
        ; (= (fuel-used) 0)
	; (= (fuel) 7.0)
)
(:goal (and
        (>= (cash) 1000)
))
;(:metric minimize (fuel-used)) 
)
