# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 07:31:20 2024

@author: ASUS
"""

import numpy as np
import random
import math
from itertools import permutations

class Customer:
    def __init__(self, id, demand, service_time, time_window):
        self.id = id
        self.demand = demand
        self.service_time = service_time
        self.time_window = time_window

class Depot:
    def __init__(self, id, location):
        self.id = id
        self.location = location
        self.customers = []

class RouteInfo:
    def __init__(self, route, distance, load, total_time):
        self.route = route
        self.distance = distance
        self.load = load
        self.total_time = total_time

def assign_customers_to_depots(depots, customers, time_matrix):
    # Reset depot customers
    for depot in depots:
        depot.customers = []

    # Assign each customer to the closest depot based on travel time
    for customer in customers:
        min_time = float('inf')
        best_depot = None

        for depot in depots:
            travel_time = time_matrix[depot.id][customer.id]
            if travel_time < min_time:
                min_time = travel_time
                best_depot = depot

        if best_depot:
            best_depot.customers.append(customer)

def calculate_route_details(route, distance_matrix, time_matrix, customers, capacity):
    if not route or len(route) < 2:
        return None

    # Calculate total demand first
    total_load = 0
    for i in range(1, len(route)-1):  # Skip first and last nodes (depot)
        customer = next((c for c in customers if c.id == route[i]), None)
        if not customer:
            return None
        total_load += customer.demand

    if total_load > capacity:
        return None

    # Calculate total time and distance
    total_time = 0
    total_distance = 0
    current_time = 8  # Start time at 8:00

    # Calculate travel times, distances and service times
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]

        # Add travel time and distance between nodes
        travel_time = time_matrix[from_node][to_node]
        travel_distance = distance_matrix[from_node][to_node]

        total_time += travel_time
        total_distance += travel_distance
        current_time += travel_time

        # If this is a customer node, add service time
        if i > 0 and i < len(route) - 1:  # Skip first and last nodes (depot)
            customer = next((c for c in customers if c.id == route[i]), None)
            if customer:
                total_time += customer.service_time
                current_time += customer.service_time

        # Check time window constraints
        if i < len(route) - 1:  # Don't check for return to depot
            next_customer = next((c for c in customers if c.id == to_node), None)
            if next_customer and hasattr(next_customer, 'time_window'):
                if current_time > next_customer.time_window[1]:
                    return None

    # Check if total time exceeds 7 hours
    if total_time > 7:
        return None

    return RouteInfo(route, total_distance, total_load, total_time)

def check_constraints(route, distance_matrix, time_matrix, capacity, customers):
    route_info = calculate_route_details(route, distance_matrix, time_matrix, customers, capacity)
    return route_info is not None

def sequential_insertion(depot, customers, distance_matrix, time_matrix, capacity):
    """
    Sequential Insertion algorithm for VRP
    - Starts from depot
    - Selects nearest unrouted customer
    - Checks capacity (≤ 38) and time (≤ 7) constraints
    """
    routes = []
    unrouted_customers = list(depot.customers)

    while unrouted_customers:
        # Start a new route
        current_route = [depot.id]
        current_load = 0
        total_time = 0
        total_distance = 0

        # Find nearest customer to depot
        while unrouted_customers:
            best_next = None
            min_distance = float('inf')

            for customer in unrouted_customers:
                if current_route[-1] == depot.id:
                    # Calculate distance from depot to customer
                    distance = distance_matrix[depot.id][customer.id]
                else:
                    # Calculate distance from last customer to current customer
                    distance = distance_matrix[current_route[-1]][customer.id]

                if distance < min_distance:
                    # Check if adding this customer is feasible
                    test_route = current_route + [customer.id, depot.id]
                    route_info = calculate_route_details(test_route, distance_matrix, time_matrix, customers, capacity)

                    if route_info and route_info.total_time <= 7 and current_load + customer.demand <= capacity:
                        min_distance = distance
                        best_next = customer

            if best_next:
                current_route.append(best_next.id)
                current_load += best_next.demand
                total_distance += min_distance
                total_time += time_matrix[current_route[-2]][best_next.id] + best_next.service_time
                unrouted_customers.remove(best_next)
            else:
                break

        # Ensure the route returns to the depot
        current_route.append(depot.id)

        # Evaluate different route configurations
        possible_routes = []
        for i in range(1, len(current_route) - 1):
            for j in range(i + 1, len(current_route)):
                new_route = current_route[:i] + current_route[i:j][::-1] + current_route[j:]
                route_info = calculate_route_details(new_route, distance_matrix, time_matrix, customers, capacity)
                if route_info and route_info.total_time <= 7:
                    possible_routes.append((new_route, route_info.distance))

        # Choose the best route configuration
        if possible_routes:
            best_route = min(possible_routes, key=lambda x: x[1])[0]
            route_info = calculate_route_details(best_route, distance_matrix, time_matrix, customers, capacity)
            routes.append(route_info)
        else:
            # Complete the route
            route_info = calculate_route_details(current_route, distance_matrix, time_matrix, customers, capacity)
            if route_info:
                routes.append(route_info)

    return routes

def simulated_annealing(depot, distance_matrix, time_matrix, capacity, customers,
                        initial_temperature=20, cooling_rate=0.9, max_iterations=10, maxsukses=5):
    """
    Simulated annealing algorithm with Josephus permutation logic
    """
    initial_routes = sequential_insertion(depot, customers, distance_matrix, time_matrix, capacity)
    
    if not initial_routes:
        return None, 0, 0

    current_sequence = []
    for route in initial_routes:
        current_sequence.extend(route.route[1:-1])  # Exclude depot nodes
    
    best_sequence = current_sequence[:]
    best_routes = initial_routes
    best_distance = sum(route.distance for route in initial_routes)
    temperature = initial_temperature
    success_count = 0

    def evaluate_solution(sequence):
        if not sequence:
            return [], float('inf'), float('inf')
            
        routes = []
        current_route = [depot.id]
        current_load = 0
        total_distance = 0
        total_time = 0
        
        for customer_id in sequence:
            customer = next((c for c in customers if c.id == customer_id), None)
            if customer:
                test_route = current_route + [customer_id, depot.id]
                route_info = calculate_route_details(test_route, distance_matrix, time_matrix, customers, capacity)
                
                if route_info and current_load + customer.demand <= capacity and route_info.total_time <= 7:
                    current_route.append(customer_id)
                    current_load += customer.demand
                else:
                    if len(current_route) > 1:
                        current_route.append(depot.id)
                        route_info = calculate_route_details(current_route, distance_matrix, time_matrix, customers, capacity)
                        if route_info:
                            routes.append(route_info)
                            total_distance += route_info.distance
                            total_time += route_info.total_time
                    
                    current_route = [depot.id, customer_id]
                    current_load = customer.demand

        if len(current_route) > 1:
            current_route.append(depot.id)
            route_info = calculate_route_details(current_route, distance_matrix, time_matrix, customers, capacity)
            if route_info:
                routes.append(route_info)
                total_distance += route_info.distance
                total_time += route_info.total_time
                
        return routes, total_distance, total_time

    # Main simulated annealing loop
    for temp_iteration in range(max_iterations):
        temperature = initial_temperature * (cooling_rate ** temp_iteration)
        
        for iteration in range(max_iterations):
            if len(current_sequence) <= 1:
                break
                
            N1 = random.randint(0, len(current_sequence)-1)
            N2 = random.randint(N1+1, len(current_sequence))
            
            front = current_sequence[:N1]
            middle = current_sequence[N1:N2]
            back = current_sequence[N2:]
            
            r = random.random()
            
            if r < 0.5:
                middle = middle[::-1]
                new_sequence = front + middle + back
            else:
                temp = front + back
                if temp:
                    R2 = random.randint(0, len(temp))
                    new_sequence = temp[:R2] + middle + temp[R2:]

            new_routes, new_distance, new_time = evaluate_solution(new_sequence)
            delta_E = new_distance - best_distance
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / temperature):
                best_sequence = new_sequence
                best_routes = new_routes
                best_distance = new_distance
                success_count += 1

            if success_count >= maxsukses:
                break

        if temp_iteration >= max_iterations - 1:
            break

    return best_routes, best_distance, sum(route.total_time for route in best_routes)

def shift_one_zero(routes, distance_matrix, time_matrix, capacity, customers):
    """Shift(1,0) operator: moves 1 customer from one route to another"""
    best_routes = routes.copy()
    best_total_distance = sum(route.distance for route in routes)
    
    for i, source_route in enumerate(routes):
        for j, target_route in enumerate(routes):
            if i != j:
                for pos1 in range(1, len(source_route.route)-1):
                    customer = source_route.route[pos1]
                    for pos2 in range(1, len(target_route.route)):
                        # Create new routes
                        new_source = source_route.route[:pos1] + source_route.route[pos1+1:]
                        new_target = target_route.route[:pos2] + [customer] + target_route.route[pos2:]

                        # Validate both routes
                        source_info = calculate_route_details(new_source, distance_matrix, time_matrix, customers, capacity)
                        target_info = calculate_route_details(new_target, distance_matrix, time_matrix, customers, capacity)

                        if source_info and target_info:
                            new_routes = routes.copy()
                            new_routes[i] = source_info
                            new_routes[j] = target_info
                            new_total_distance = sum(route.distance for route in new_routes)

                            if new_total_distance < best_total_distance:
                                best_routes = new_routes
                                best_total_distance = new_total_distance

    return best_routes

def swap_two_two(routes, distance_matrix, time_matrix, capacity, customers):
    """Swap(2,2) operator: swaps 2 customers between routes"""
    best_routes = routes.copy()
    best_total_distance = sum(route.distance for route in routes)

    for i, route1 in enumerate(routes):
        for j, route2 in enumerate(routes):
            if i != j:
                for p1 in range(1, len(route1.route)-2):
                    for p2 in range(1, len(route2.route)-2):
                        # Get pairs of customers
                        customers1 = route1.route[p1:p1+2]
                        customers2 = route2.route[p2:p2+2]

                        # Create new routes
                        new_route1 = route1.route[:p1] + customers2 + route1.route[p1+2:]
                        new_route2 = route2.route[:p2] + customers1 + route2.route[p2+2:]

                        # Validate routes
                        info1 = calculate_route_details(new_route1, distance_matrix, time_matrix, customers, capacity)
                        info2 = calculate_route_details(new_route2, distance_matrix, time_matrix, customers, capacity)

                        if info1 and info2:
                            new_routes = routes.copy()
                            new_routes[i] = info1
                            new_routes[j] = info2
                            new_total_distance = sum(route.distance for route in new_routes)

                            if new_total_distance < best_total_distance:
                                best_routes = new_routes
                                best_total_distance = new_total_distance

    return best_routes

def swap_one_one(routes, distance_matrix, time_matrix, capacity, customers):
    """Swap(1,1) operator: swaps 1 customer between routes"""
    best_routes = routes.copy()
    best_total_distance = sum(route.distance for route in routes)

    for i, route1 in enumerate(routes):
        for j, route2 in enumerate(routes):
            if i != j:
                for pos1 in range(1, len(route1.route)-1):
                    for pos2 in range(1, len(route2.route)-1):
                        # Swap customers
                        new_route1 = route1.route.copy()
                        new_route2 = route2.route.copy()
                        new_route1[pos1], new_route2[pos2] = new_route2[pos2], new_route1[pos1]
                        
                        # Validate routes
                        info1 = calculate_route_details(new_route1, distance_matrix, time_matrix, customers, capacity)
                        info2 = calculate_route_details(new_route2, distance_matrix, time_matrix, customers, capacity)

                        if info1 and info2:
                            new_routes = routes.copy()
                            new_routes[i] = info1
                            new_routes[j] = info2
                            new_total_distance = sum(route.distance for route in new_routes)

                            if new_total_distance < best_total_distance:
                                best_routes = new_routes
                                best_total_distance = new_total_distance

    return best_routes

def cross_exchange(routes, distance_matrix, time_matrix, capacity, customers):
    """Cross operator: cross exchange between routes"""
    best_routes = routes.copy()
    best_total_distance = sum(route.distance for route in routes)

    for i, route1 in enumerate(routes):
        for j, route2 in enumerate(routes):
            if i != j:
                for p1 in range(1, len(route1.route)-1):
                    for p2 in range(1, len(route2.route)-1):
                        # Cross exchange
                        new_route1 = route1.route[:p1] + route2.route[p2:]
                        new_route2 = route2.route[:p2] + route1.route[p1:]

                        # Validate routes
                        info1 = calculate_route_details(new_route1, distance_matrix, time_matrix, customers, capacity)
                        info2 = calculate_route_details(new_route2, distance_matrix, time_matrix, customers, capacity)

                        if info1 and info2:
                            new_routes = routes.copy()
                            new_routes[i] = info1
                            new_routes[j] = info2
                            new_total_distance = sum(route.distance for route in new_routes)

                            if new_total_distance < best_total_distance:
                                best_routes = new_routes
                                best_total_distance = new_total_distance

    return best_routes

def exchange_within_route(routes, distance_matrix, time_matrix, capacity, customers):
    """Exchange operator: swaps positions within same route"""
    best_routes = routes.copy()
    best_total_distance = sum(route.distance for route in routes)

    for i, route in enumerate(routes):
        for pos1 in range(1, len(route.route) - 1):
            for pos2 in range(pos1 + 1, len(route.route) - 1):
                new_route = route.route.copy()
                new_route[pos1], new_route[pos2] = new_route[pos2], new_route[pos1]

                new_route_info = calculate_route_details(new_route, distance_matrix, time_matrix, customers, capacity)
                if new_route_info:
                    new_routes = routes.copy()
                    new_routes[i] = new_route_info
                    new_total_distance = sum(r.distance for r in new_routes)

                    if new_total_distance < best_total_distance:
                        best_routes = new_routes
                        best_total_distance = new_total_distance

    return best_routes

def rvnd(routes, distance_matrix, time_matrix, capacity, customers):
    """Random Variable Neighborhood Descent (RVND)"""
    neighborhoods = [shift_one_zero, exchange_within_route, swap_two_two, swap_one_one, cross_exchange]
    random.shuffle(neighborhoods)
    improved = True
    
    while improved:
        improved = False
        for operator in neighborhoods:
            new_solution = operator(routes, distance_matrix, time_matrix, capacity, customers)
            if new_solution:
                new_cost = sum(route.distance for route in new_solution)
                current_cost = sum(route.distance for route in routes)
                if new_cost < current_cost:
                    routes = new_solution
                    improved = True
                    break
    return routes


def gvns(depot, distance_matrix, time_matrix, capacity, customers):
    """
    General Variable Neighborhood Search (GVNS) implementation
    """
    def calculate_total_cost(routes):
        return sum(route.distance for route in routes)
    
    # Initial solution
    current_solution = sequential_insertion(depot, customers, distance_matrix, time_matrix, capacity)
    if not current_solution:
        return None
    
    best_solution = current_solution
    best_cost = calculate_total_cost(best_solution)
    
    k_max = 4  # Number of neighborhood structures
    max_iterations = 100
    max_no_improve = 20
    no_improve = 0
    
    def perturbation(solution, level, distance_matrix, time_matrix, capacity, customers):
        """Perturbation step using Shift (1,0) operator"""
        for _ in range(level):
            solution = shift_one_zero(solution, distance_matrix, time_matrix, capacity, customers)
        return solution

    while no_improve < max_no_improve:
        k = 0
        while k < k_max:
            # Perturbation
            x_prime = perturbation(best_solution, k + 1, distance_matrix, time_matrix, capacity, customers)
            
            # Local search with RVND
            x_double_prime = rvnd(x_prime, distance_matrix, time_matrix, capacity, customers)
            
            # Move or not
            new_cost = calculate_total_cost(x_double_prime)
            if new_cost < best_cost:
                best_solution = x_double_prime
                best_cost = new_cost
                k = 0  # Reset neighborhood structure
                no_improve = 0
            else:
                k += 1
        
        no_improve += 1
    
    return best_solution, best_cost, sum(route.total_time for route in best_solution)

def format_routes_output(routes, start_index=1):
    output = []
    route_number = start_index
    for route in routes:
        route_str = ' – '.join(map(str, route.route))
        output.append(f"Rute {route_number} : {route_str}\nTotal kapasitas : {route.load}\nTotal waktu : {route.total_time:.4f}\nTotal jarak : {route.distance:.1f}\n")
        route_number += 1
    return '\n'.join(output)

def print_gvns_results(result, distance_matrix=None, time_matrix=None, customers=None, capacity=None):
    """Print GVNS results in required format"""
    if isinstance(result, tuple):
        routes, total_distance, total_time = result
    else:
        routes = result
        total_distance = sum(route.distance for route in routes)
        total_time = sum(route.total_time for route in routes)

    for i, route_info in enumerate(routes, 1):
        if isinstance(route_info, RouteInfo):
            print(f"Rute {i} : {' - '.join(map(str, route_info.route))}")
            print(f"Total kapasitas : {route_info.load}")
            print(f"Total waktu : {route_info.total_time:.4f}")
            print(f"Total jarak : {route_info.distance}")
        elif all(x is not None for x in [distance_matrix, time_matrix, customers, capacity]):
            # If route_info is just a list, calculate route details
            route_details = calculate_route_details(route_info, distance_matrix, time_matrix, customers, capacity)
            if route_details:
                print(f"Rute {i} : {' - '.join(map(str, route_details.route))}")
                print(f"Total kapasitas : {route_details.load}")
                print(f"Total waktu : {route_details.total_time:.4f}")
                print(f"Total jarak : {route_details.distance}")
        else:
            # If we don't have the necessary parameters, just print the route
            print(f"Rute {i} : {' - '.join(map(str, route_info))}")
        print()

    print(f"Total keseluruh waktu (semua rute) : {total_time:.4f}")
    print(f"Total keseluruhan jarak (semua rute) : {total_distance}")

def print_sa_results(routes, total_distance, total_time):
    """
    Print the results from simulated annealing algorithm
    """
    if not routes:
        for i in range(1, 3):  # Print 2 empty routes
            print(f"Rute {i} : -")
            print("Total kapasitas : -")
            print("Total waktu : -")
            print("Total jarak : -")
            print()
        print("Total keseluruh waktu (semua rute) : -")
        print("Total keseluruhan jarak (semua rute) : -")
        return

    for i, route in enumerate(routes, 1):
        print(f"Rute {i} : {' - '.join(map(str, route.route))}")
        print(f"Total kapasitas : {route.load}")
        print(f"Total waktu : {route.total_time:.4f}")
        print(f"Total jarak : {route.distance:.1f}")
        print()

    print(f"Total keseluruh waktu (semua rute) : {total_time:.4f}")
    print(f"Total keseluruhan jarak (semua rute) : {total_distance:.1f}")

def main():
    """
    Main function to execute the vehicle routing problem solution using sequential insertion,
    simulated annealing, and general variable neighborhood search (GVNS) algorithms.
    
    It takes input for depots, customers, vehicle capacity, and distance and time matrices.
    The function assigns customers to depots and computes routes for each depot using the 
    sequential insertion algorithm. It further optimizes the routes using simulated annealing 
    and GVNS, printing out the results for each optimization step.
    """
    # Input data
    num_depots = int(input("Masukkan jumlah depot: "))
    num_customers = int(input("Masukkan jumlah pelanggan: "))
    capacity = int(input("Masukkan kapasitas kendaraan: "))

    # Input matriks jarak
    print("Masukkan matriks jarak (depot ke depot, depot ke pelanggan, pelanggan ke pelanggan):")
    distance_matrix = np.zeros((num_depots + num_customers, num_depots + num_customers))
    for i in range(num_depots + num_customers):
        row = input(f"Masukkan jarak untuk {'Depot' if i < num_depots else 'Pelanggan'} {i} (pisahkan dg spasi): ")
        distance_matrix[i] = list(map(float, row.split()))

    # Input matriks waktu
    print("Masukkan matriks waktu (depot ke depot, depot ke pelanggan, pelanggan ke pelanggan):")
    time_matrix = np.zeros((num_depots + num_customers, num_depots + num_customers))
    for i in range(num_depots + num_customers):
        row = input(f"Masukkan waktu untuk {'Depot' if i < num_depots else 'Pelanggan'} {i} (pisahkan dg spasi): ")
        time_matrix[i] = list(map(float, row.split()))

    # Input data pelanggan
    customers = []
    demand_list = []
    service_time_list = []

    for i in range(num_customers):
        demand = int(input(f"Masukkan permintaan untuk pelanggan {i}: "))
        demand_list.append(demand)
        service_time = float(input(f"Masukkan waktu pelayanan untuk pelanggan {i} (dalam float): "))
        service_time_list.append(service_time)

    # Input waktu jendela untuk semua pelanggan
    time_window_start = int(input("Masukkan waktu jendela awal untuk semua pelanggan: "))
    time_window_end = int(input("Masukkan waktu jendela akhir untuk semua pelanggan: "))

    for i in range(num_customers):
        customers.append(Customer(i + num_depots, demand_list[i], service_time_list[i], (time_window_start, time_window_end)))

    # Create depots
    depots = [Depot(i, (0, 0)) for i in range(num_depots)]

    # Assign customers to depots
    assign_customers_to_depots(depots, customers, time_matrix)

    # Print daftar pelanggan untuk setiap depot
    for depot in depots:
        customer_ids = [customer.id for customer in depot.customers]
        print(f"Depot {depot.id} memiliki pelanggan: {customer_ids}")

    # Initialize data structures
    all_routes = []
    total_overall_time = 0
    total_overall_distance = 0

    print("\n--- Sequential Insertion ---")

    # Process each depot
    for depot in depots:
        depot_routes = sequential_insertion(depot, customers, distance_matrix, time_matrix, capacity)
        all_routes.extend(depot_routes)

    # Print all routes in order
    for idx, route in enumerate(all_routes):
        route_str = " – ".join(str(node) for node in route.route)
        print(f"Rute {idx + 1} : {route_str}")
        print(f"Total kapasitas : {route.load}")
        print(f"Total waktu : {route.total_time:.4f}")
        print(f"Total jarak : {route.distance:.1f}")
        print()

        total_overall_time += route.total_time
        total_overall_distance += route.distance

    print(f"Total keseluruh waktu (semua rute) : {total_overall_time:.4f}")
    print(f"Total keseluruhan jarak (semua rute) : {total_overall_distance:.1f}")

    # Simulated Annealing
    print("\n--- Simulated Annealing ---")
    sa_total_time = 0
    sa_total_distance = 0
    
    route_number = 1
    for depot in depots:
        routes, distance, time = simulated_annealing(depot, distance_matrix, time_matrix, capacity, depot.customers)
        if routes and any(len(r.route) > 2 for r in routes):  # Only count non-empty routes
            print(format_routes_output(routes, start_index=route_number))
            route_number += len(routes)
            sa_total_time += time
            sa_total_distance += distance
        else:
            print(format_routes_output([], start_index=route_number))  # Print empty routes format
            
    print("\nTotal keseluruhan waktu (semua depot) : {:.4f}".format(sa_total_time))
    print("Total keseluruhan jarak (semua depot) : {:.1f}".format(sa_total_distance))

    # GVNS
    print("\n--- GVNS ---")
    gvns_total_time = 0
    gvns_total_distance = 0
    
    route_number = 1
    for depot in depots:
        result = gvns(depot, distance_matrix, time_matrix, capacity, depot.customers)
        if result:
            routes, best_distance, best_time = result
            print(format_routes_output(routes, start_index=route_number))
            route_number += len(routes)
            gvns_total_time += best_time
            gvns_total_distance += best_distance
            
    print("\nTotal keseluruhan waktu (semua depot) : {:.4f}".format(gvns_total_time))
    print("Total keseluruhan jarak (semua depot) : {:.1f}".format(gvns_total_distance))

if __name__ == "__main__":
    main()