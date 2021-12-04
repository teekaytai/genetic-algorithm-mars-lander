from collections import namedtuple
import math
import random
from time import process_time

MAP_WIDTH = 7000
MAP_HEIGHT = 3000
ZONE_WIDTH = 350  # Divide map into vertical stripe zones to limit number of collision checks needed

MAX_ANGLE_CHANGE = 15
MAX_POW_CHANGE = 1
MAX_LANDING_VX = 20
MAX_LANDING_VY = 40

SINE = [math.sin(math.radians(d)) for d in range(0, 91)]  # Look-up table for sine values

# Possible lander statuses
FLYING = 0
CRASHED = 1
CRASHED_ON_LANDING_SITE = 2
LANDED_SAFELY = 3

CHROM_LENGTH = 200
POP_SIZE = 80
ELITES_NUM = int(0.2 * POP_SIZE)  # How many chromosomes directly selected for next generation
CROSSOVERS_NUM = (POP_SIZE - ELITES_NUM) // 2  # How many crossovers done per generation (each yields 2 chromosomes)
MUTATION_PROBABILTY = 0.25  # Probability of chromosome mutating

Point = namedtuple('Point', ['x', 'y'])
LanderState = namedtuple('LanderState', ['p', 'vx', 'vy', 'fuel', 'angle', 'power'])


# A gene consists of a change in angle between -15 and 15 and a change in power between -1 and 1
def random_gene(gene=None):
    if gene is None:
        gene = [0, 0]
    gene[0] = random.randint(-MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE)
    gene[1] = random.randint(-MAX_POW_CHANGE, MAX_POW_CHANGE)
    return gene


def random_chrom():
    return [random_gene() for _ in range(CHROM_LENGTH)]


def create_population():
    return [random_chrom() for _ in range(POP_SIZE)]


# Returns a positive number if points a, b, c are in counter-clockwise order, a negative number if they are clockwise and 0 if they are collinear
def ccw(a, b, c):
    return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)


# Returns True if line segments ab and cd intersect
def intersect(a, b, c, d):
    return ccw(a,b,c) * ccw(a,b,d) <= 0 and ccw(c,d,a) * ccw(c,d,b) <= 0


def dist(object_1, object_2):
    return math.dist((object_1.x, object_1.y), (object_2.x, object_2.y))


def simulate(chrom, lander_state):
    p, vx, vy, fuel, angle, power = lander_state
    zone = p.x // ZONE_WIDTH
    lander_status = FLYING
    for turn, (angle_change, pow_change) in enumerate(chrom):
        angle = min(max(angle + angle_change, -90), 90)
        power = min(max(power + pow_change, 0), 4, fuel)
        
        abs_angle = abs(angle)
        delta_vx = math.copysign(power * SINE[abs_angle], -angle)
        delta_vy = power * SINE[90 - abs_angle] - 3.711
        vx += delta_vx
        vy += delta_vy
        X = p.x + vx
        Y = p.y + vy
        q = Point(X, Y)
        fuel -= power
        next_zone = int(X // ZONE_WIDTH)

        if X < 0 or X >= MAP_WIDTH or Y < 0 or Y > MAP_HEIGHT:
            lander_status = CRASHED
            break

        # Check for collisions with surface segments within zone or within two zones if border crossed
        if next_zone != zone:
            collided_segment = next((segment for segment in CROSS_ZONE_SURFACES[min(zone, next_zone)] if intersect(p, q, *segment)), None)
        else:
            collided_segment = next((segment for segment in ZONE_SURFACES[zone] if intersect(p, q, *segment)), None)

        if collided_segment:
            if collided_segment is LANDING_SEGMENT:
                if angle == 0 and abs(vx) <= MAX_LANDING_VX and abs(vy) <= MAX_LANDING_VY:
                    lander_status = LANDED_SAFELY
                else:
                    lander_status = CRASHED_ON_LANDING_SITE
            else:
                lander_status = CRASHED
            break
        
        p = q
        zone = next_zone
    
    return LanderState(p, vx, vy, fuel, angle, power), lander_status, turn


def get_score(chrom, lander_state):
    end_lander_state, end_lander_status, genes_used = simulate(chrom, lander_state)
    score = (8000 - dist(end_lander_state.p, LANDING_SITE_CENTRE)) / 800  # More points for finishing nearer to landing site
    if end_lander_status == CRASHED or end_lander_status == FLYING:
        score += genes_used * 0.05  # More points for surviving longer
    elif end_lander_status == CRASHED_ON_LANDING_SITE:
        score += 50
        score -= max(abs(end_lander_state.vx) - MAX_LANDING_VX, 0) / MAX_LANDING_VX * 2  # Penalise excess speed when landing
        score -= max(abs(end_lander_state.vy) - MAX_LANDING_VY, 0) / MAX_LANDING_VY * 6  # Penalise excess speed when landing
        score -= abs(end_lander_state.angle) / 90  # Penalise excess angle when landing
    elif end_lander_status == LANDED_SAFELY:
        score += 100
        score += end_lander_state.fuel  # More fuel remaining is better
    return score


# Use fitness proportionate selection to pick 2 different parents
def choose_random_chrom(population, cumulative_scores, total_score):
    random_value = random.random() * total_score
    for i in range(POP_SIZE):
        if random_value <= cumulative_scores[i]:
            return population[i]


# One-point crossover: Choose random crossover point and swap genes on the right
def crossover(parent1, parent2, child1, child2):
    random_index = random.randint(1, CHROM_LENGTH - 2)
    for i in range(random_index):
        child1_gene = child1[i]
        child2_gene = child2[i]
        child1_gene[0], child1_gene[1] = parent1[i]
        child2_gene[0], child2_gene[1] = parent2[i]
    for i in range(random_index, CHROM_LENGTH):
        child1_gene = child1[i]
        child2_gene = child2[i]
        child1_gene[0], child1_gene[1] = parent2[i]
        child2_gene[0], child2_gene[1] = parent1[i]


def mutate(chrom):
    if random.random() < MUTATION_PROBABILTY:
        # Replace one gene with a random gene
        mutating_gene = chrom[random.randint(0, CHROM_LENGTH - 1)]
        random_gene(mutating_gene)


# Overwrite provided next_population array to reduce array creations and speed up generation
def generation(population, lander_state, next_population, past_scores=None):
    if past_scores:
        # Carry over scores of elite chromosomes carried over from last generation
        scores = past_scores[:ELITES_NUM] + [get_score(chrom, lander_state) for chrom in population[ELITES_NUM:]]
    else:
        scores = [get_score(chrom, lander_state) for chrom in population]
    
    # Sort scores in descending order
    scores, population = (list(t) for t in zip(*sorted(zip(scores, population), reverse = True)))
    
    cumulative_scores = []
    running_total = 0
    for score in scores:
        running_total += score
        cumulative_scores.append(running_total)
    total_score = running_total

    # Copy over elite chromosomes without modification
    for i in range(ELITES_NUM):
        parent = population[i]
        child = next_population[i]
        for j in range(CHROM_LENGTH):
            child[j][0], child[j][1] = parent[j]

    # Select parents and do crossovers to create remaining children
    for i in range(ELITES_NUM, POP_SIZE, 2):
        parent1 = choose_random_chrom(population, cumulative_scores, total_score)
        while True:
            # Ensure parents different
            parent2 = choose_random_chrom(population, cumulative_scores, total_score)
            if parent1 is not parent2:
                break
        child1 = next_population[i]
        child2 = next_population[i+1]
        crossover(parent1, parent2, child1, child2)
        mutate(child1)
        mutate(child2)
    
    return scores  # Save scores so that elite scores can be reused later


# Save surface of Mars, grouping segments by vertical striped zones
surface_n = int(input())
ZONE_SURFACES = [set() for _ in range(MAP_WIDTH // ZONE_WIDTH)]
LANDING_SEGMENT = None
LANDING_SITE_CENTRE = None
prev_x, prev_y = [int(j) for j in input().split()]
prev_zone = prev_x // ZONE_WIDTH
for i in range(surface_n - 1):
    land_x, land_y = [int(j) for j in input().split()]
    segment = (Point(prev_x, prev_y), Point(land_x, land_y))
    segment_zone = land_x // ZONE_WIDTH
    
    for j in range(min(prev_zone, segment_zone), max(prev_zone, segment_zone) + 1):
        ZONE_SURFACES[j].add(segment)

    if prev_y == land_y:
        LANDING_SEGMENT = segment
        x = (prev_x + land_x) / 2
        LANDING_SITE_CENTRE = Point(x, land_y)
    
    prev_x = land_x
    prev_y = land_y
    prev_zone = segment_zone

# Each entry stores segments from two adjacent zones. For testing collisions when crossing zone border
CROSS_ZONE_SURFACES = [ZONE_SURFACES[i].union(ZONE_SURFACES[i+1]) for i in range(0, len(ZONE_SURFACES)-1)]


population = None
next_population = create_population()  # These entries don't matter, they will be overwritten

while True:
    x, y, *inputs = [int(i) for i in input().split()]
    p = Point(x, y)
    lander_state = LanderState(p, *inputs)
    t_start = process_time()

    if population is None:
        population = create_population()

        # Help teach bot how to fly upwards at slight angle. Bot struggles to find this on its own
        fly_up_left_chrom = [[0, 1] for _ in range(CHROM_LENGTH)]  # Fly at 5 degree angle at power 4
        target_angle = 5
        lander_angle = lander_state.angle
        i = 0
        while lander_angle != target_angle:
            angle_change = min(target_angle - lander_angle, MAX_ANGLE_CHANGE) if lander_angle < target_angle else max(target_angle - lander_angle, -MAX_ANGLE_CHANGE)
            fly_up_left_chrom[i][0] = angle_change
            lander_angle += angle_change
            i += 1
        
        fly_up_right_chrom = [[0, 1] for _ in range(CHROM_LENGTH)]  # Fly at -5 degree angle at power 4
        target_angle = -5
        lander_angle = lander_state.angle
        i = 0
        while lander_angle != target_angle:
            angle_change = min(target_angle - lander_angle, MAX_ANGLE_CHANGE) if lander_angle < target_angle else max(target_angle - lander_angle, -MAX_ANGLE_CHANGE)
            fly_up_right_chrom[i][0] = angle_change
            lander_angle += angle_change
            i += 1

        population[0] = fly_up_left_chrom
        population[1] = fly_up_right_chrom
    else:
        population = [chrom[1:] + [random_gene()] for chrom in population]  # Carry over chromosomes from previous round with their first genes removed
    
    scores = None
    t_elapsed = process_time() - t_start
    while True:
        if scores:
            scores = generation(population, lander_state, next_population, scores)
        else:
            scores = generation(population, lander_state, next_population)
        population, next_population = next_population, population
        t_elapsed2 = process_time() - t_start
        if t_elapsed2 * 2 - t_elapsed >= 0.1:
            # Next generation probably will not finish in time.
            break
        t_elapsed = t_elapsed2

    best_gene = population[0][0]
    angle_out = min(max(lander_state.angle + best_gene[0], -90), 90)
    power_out = min(max(lander_state.power + best_gene[1], 0), 4)
    print(angle_out, power_out)
