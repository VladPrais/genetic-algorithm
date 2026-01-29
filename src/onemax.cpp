#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "ga.h"

typedef std::vector<int> GeneType;
typedef int FitnessType;
typedef ga::BaseIndividual<GeneType, FitnessType> Individual;

std::uniform_int_distribution dist(0, 1);

const size_t LEN_GENES = 6e2;

// The function generates the one Individual. Will be used in Genetic Algorithm to create initial population.
Individual generator(std::mt19937 &engine)
{
	GeneType v(LEN_GENES);

	std::generate(v.begin(), v.end(), [&engine](){ return dist(engine); });

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
	if(i.fitness >= LEN_GENES)
		return true;
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
	int pop_size = 1e4, max_generations = 300, elite = 50;
	double cxpb = 0.6, mtpb = 0.2, mgpb = 0.01;
	int tourn_size = 4;

	ga::tournament<Individual> selection(tourn_size);
	ga::crossover<Individual> crossover(cxpb, ga::cx_one_point<Individual>());
	ga::mutation<Individual> mutation(mtpb, mgpb, ga::mut_inverse<Individual>());

	ga::ParallelGenetic<GeneType, FitnessType> g_alg(generator, comparator, evaluate, stop_cond, selection, crossover, mutation, max_generations, pop_size, elite);

	ga::GeneticOutput gen_out = g_alg();
	std::cout << gen_out.best_ever << std::endl;

	return 0;
}
