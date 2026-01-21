#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "ga.h"

typedef std::vector<int> GeneType;
typedef int FitnessType;
typedef ga::BaseIndividual<GeneType, FitnessType> Individual;

std::random_device rd;
std::mt19937 engine(rd());
std::uniform_int_distribution dist(0, 1);

const size_t LEN_GENES = 100;

// The function generates the one Individual. Will be used in Genetic Algorithm to create initial population.
Individual generator(void)
{
	GeneType v(LEN_GENES);

	std::generate(v.begin(), v.end(), [](){ return dist(engine); });

	Individual I(v);

	return I;
}

// The function to get fitness value.
FitnessType evaluate(Individual &i)
{
	return std::accumulate(i.genes.begin(), i.genes.end(), 0);
}

// Stop conditions. True if stop else False.
bool stop_cond(Individual& i)
{
	if(i.fitness == LEN_GENES)
	{
		return true;
	}
	return false;
}

std::ostream& operator<<(std::ostream &stream, const Individual &i)
{
	for(auto g: i.genes)
	{
		stream << g << ' ';
	}
	stream << "  " << i.fitness;

	return stream;
}

bool comparator(const Individual &lhs, const Individual &rhs)
{
	return lhs.fitness > rhs.fitness;
}

int main(void)
{
	int pop_size = 300, max_generations = 50, elite = 100;
	double cxpb = 0.7, mtpb = 0.2, mgpb = 0.05;
	int tourn_size = 6;

	ga::tournament<Individual> sel(tourn_size);
	ga::crossover<Individual> crossover(cxpb, ga::cx_one_point<Individual>);
	ga::mutation<Individual> mutation(mtpb, mgpb, ga::inverse_mut<Individual>());

	ga::GeneticAlgorithm<GeneType, FitnessType> g_alg(max_generations, pop_size, elite, generator, comparator, evaluate, stop_cond, sel);

	Individual v = g_alg(crossover, mutation);

	std::cout << v << std::endl;

	return 0;
}
