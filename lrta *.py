from itertools import combinations_with_replacement, permutations
from functools import reduce


class Person:
    def __init__(self, name, initial_delay, travel_time):
        self.name = name
        self.time_at_coffee = 0
        self.time_at_cinema = 0
        self.start_time = 15
        self.initial_delay = initial_delay
        self.total_delay = initial_delay
        self.travel_time = travel_time
        self.is_at = 'Home'
        self.is_inside_coffee = False
        self.is_inside_cinema = False
        self.finished_coffee = False
        self.watched_movie = False

    def add_delay(self, delay):
        self.total_delay += delay

    def compute_location_and_delay(self, current_time):
        new_delay = 0
        time_passed_since_start = current_time - self.start_time

        if self.is_at == 'Home':
            if self.initial_delay > time_passed_since_start:
                self.is_at = 'Home'
            elif (self.travel_time + self.initial_delay) > time_passed_since_start:
                self.is_at = 'Road'
            else:
                self.is_at = 'Coffee'
                new_delay = time_passed_since_start - (self.travel_time + self.initial_delay)
                self.add_delay(new_delay)
        elif self.is_at == 'Road':
            if (self.travel_time + self.initial_delay) > time_passed_since_start:
                self.is_at = 'Road'
            else:
                self.is_at = 'Coffee'
                new_delay = time_passed_since_start - (self.travel_time + self.initial_delay)
                self.add_delay(new_delay)
        elif self.is_at == 'Coffee':
            if not self.is_inside_coffee:
                new_delay = 30
                self.add_delay(new_delay)
            elif self.is_inside_coffee and self.time_at_coffee == 0:
                self.time_at_coffee += 30
            else:
                self.time_at_coffee += 30
                self.finished_coffee = True
                self.is_inside_coffee = False
                self.is_at = 'Cinema'
        elif self.is_at == 'Cinema':
            if not self.is_inside_cinema:
                new_delay = 30
                self.add_delay(new_delay)
            elif self.is_inside_cinema and self.time_at_cinema == 90:
                self.time_at_cinema += 30
                self.watched_movie = True
                self.is_inside_cinema = False
                self.is_at = 'Exit'
            else:
                self.time_at_cinema += 30

        return new_delay


class CoffeeShop:
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.current_space = max_capacity
        self.people_inside = []

    def refresh_capacity(self):
        self.people_inside = list(filter(lambda person: person.is_inside_coffee, self.people_inside))
        self.current_space = self.max_capacity - len(self.people_inside)

    def add_people(self, people):
        for person in people:
            if self.current_space > 0:
                person.is_inside_coffee = True
                self.people_inside.append(person)
                self.current_space -= 1

    @staticmethod
    def find_people_for_coffee(people):
        return list(filter(lambda person: person.is_at == 'Coffee' and not person.is_inside_coffee, people))


class Cinema:
    def __init__(self, max_capacity, movie_started):
        self.max_capacity = max_capacity
        self.current_space = max_capacity
        self.movie_started = movie_started
        self.people_inside = []

    def refresh_capacity(self):
        self.people_inside = list(filter(lambda person: person.is_inside_cinema, self.people_inside))
        self.current_space = self.max_capacity - len(self.people_inside)

    def add_people(self, people):
        for person in people:
            if self.current_space > 0:
                person.is_inside_cinema = True
                self.people_inside.append(person)
                self.current_space -= 1

    @staticmethod
    def find_people_for_cinema(people):
        return list(filter(lambda person: person.is_at == 'Cinema' and not person.is_inside_cinema, people))


class LRTAStar:
    def __init__(self):
        self.time = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450]
        self.costs = self.compute_initial_costs()
        self.people = []

    def __call__(self):
        lowest_cost_delays = [500, 500, 500, 500, 500]
        lowest_total_delay = 100000
        minimized_delay_combinations = []

        while True:
            previous_cost_delays = lowest_cost_delays
            lowest_cost_delays = self.get_min_cost_key(0)
            if previous_cost_delays == lowest_cost_delays: break
            person1 = Person('N', lowest_cost_delays[0], 30)
            person2 = Person('G', lowest_cost_delays[1], 30)
            person3 = Person('A', lowest_cost_delays[2], 10)
            person4 = Person('T', lowest_cost_delays[3], 60)
            person5 = Person('M', lowest_cost_delays[4], 60)
            self.people = [person1, person2, person3, person4, person5]
            coffee_shop = CoffeeShop(2)
            cinema1 = Cinema(3, True)
            cinema2 = Cinema(3, False)

            for time_slot in self.time:
                if self.goal_test():
                    total_delay = self.get_total_delay()
                    if total_delay < lowest_total_delay:
                        lowest_total_delay = total_delay
                    if total_delay == lowest_total_delay:
                        minimized_delay_combinations.append((lowest_cost_delays, total_delay, time_slot))
                    self.costs[0][lowest_cost_delays] = [self.costs[0][lowest_cost_delays][0], total_delay]
                    break
                else:
                    time_slot_delay = 0
                    for person in self.people:
                        time_slot_delay += person.compute_location_and_delay(time_slot)
                    coffee_shop.refresh_capacity()
                    people_need_coffee = coffee_shop.find_people_for_coffee(self.people)
                    coffee_shop.add_people(people_need_coffee)
                    if time_slot % 60 == 0:
                        people_need_cinema = cinema1.find_people_for_cinema(self.people)
                        if cinema1.movie_started:
                            cinema2.add_people(people_need_cinema)
                            cinema1.movie_started = False
                            cinema2.movie_started = True
                        else:
                            cinema1.add_people(people_need_cinema)
                            cinema2.movie_started = False
                            cinema1.movie_started = True
        print(time_slot)
        return minimized_delay_combinations

    def goal_test(self):
        for person in self.people:
            if person.finished_coffee and (person.is_inside_cinema or person.watched_movie):
                continue
            else:
                return False
        return True

    def get_total_delay(self):
        delay_per_person = list(map(lambda person: person.total_delay, self.people))
        return reduce(lambda delay, total: delay + total, delay_per_person)

    def get_min_cost_key(self, time):
        return min(self.costs[time].keys(), key=(lambda key: self.costs[time][key][0] + self.costs[time][key][1]))

    @staticmethod
    def compute_initial_costs():
        delays = list(combinations_with_replacement([0, 30, 60, 90, 120, 180, 240], 5))
        costs = {}

        for delay in delays:
            for permutation in set(permutations(delay, 5)):
                cost = reduce(lambda x, y: x + y, permutation)
                if 0 not in costs.keys(): costs.update({0: {}})
                costs[0].update({permutation: [cost, 0]})

        return costs


def print_results(best_delay_combinations):
    best_delay_combinations.pop()
    min_delay_without_initial = 10000
    min_initial_delays = []
    for best_delay_combination in best_delay_combinations:
        delays = best_delay_combination[0]
        total_initial_delay = reduce(lambda delay_person, total: delay_person + total, delays)
        delay_without_initial = best_delay_combinations[0][1] - total_initial_delay
        if delay_without_initial < min_delay_without_initial:
            min_delay_without_initial = delay_without_initial
            min_initial_delays = delays
        print('Initial Delays: N: {}, G: {}, A: {}, T: {}, M: {}'.format(delays[0], delays[1], delays[2], delays[3], delays[4]))

    print('\nBest Scores in total: {}'.format(len(best_delay_combinations)))
    print('With a minimum total delay of {} minutes'.format(best_delay_combinations[0][1]))
    print(
        'Minimum total delay without initial delays is {} minutes, for N: {}, G: {}, A: {}, T: {}, M: {}'.format(min_delay_without_initial,
                                                                                                                 min_initial_delays[0],
                                                                                                                 min_initial_delays[1],
                                                                                                                 min_initial_delays[2],
                                                                                                                 min_initial_delays[3],
                                                                                                                 min_initial_delays[4]))


if __name__ == '__main__':
    LRTA_with_problem = LRTAStar()
    best_delay_combinations = LRTA_with_problem()

    print_results(best_delay_combinations)
