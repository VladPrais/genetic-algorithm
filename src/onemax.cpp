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

// The function that allows to compare two Individuals. Crucial aspect because there is oprimization criterion (min or max).
bool comp(Individual &i1, Individual &i2)
{
	return i1.fitness < i2.fitness;
}

// The function to get fitness value.
FitnessType evaluate(Individual &i)
{
	int s = 0;

	for(auto j: i.genes)
	{
		s += j;
	}

	return s;
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

// The function that choose the one parent to mate. It will be ised in Selection operator of Genetic Algorithm
Individual sel_func(std::vector<Individual> &population)
{
	// Just one iter of Tournament Selection.
	int pop_size = population.size(), tourn_size = 3;
	std::uniform_int_distribution getter(0, pop_size - 1);
	std::vector<Individual> temp(tourn_size);

	for(int i = 0; i < tourn_size; i++)
	{
		int p = getter(engine);
		temp[i] = population[p];
	}

	Individual best = *std::max_element(temp.begin(), temp.end(), comp);

	return best;
}

// The function that mate two parents. It will be used in Crossover operator of Genetic Algorithm
void mate_func(Individual &i1, Individual &i2)
{
	int len_genes = i1.genes.size();

	std::uniform_int_distribution dot(1, len_genes - 2);
	int d = dot(engine);

	for(int i = 0; i < d; i++)
	{
		std::swap(i1.genes[i], i2.genes[i]);
	}
}

// The function that mutate one child. It will be used in Mutation operator of Genetic Algorithm
void mut_func(Individual &i)
{
	int len_genes = i.genes.size();

	std::uniform_int_distribution dot(0, len_genes - 1);

	int p = dot(engine);

	i.genes[p] = !i.genes[p];
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

int main(void)
{
	int pop_size = 300, max_generations = 50, elite = 6;
	double cxpb = 0.7, mtpb = 0.1;

	ga::GeneticAlgorithm<GeneType, FitnessType> g_alg(max_generations, pop_size, cxpb, mtpb, elite, generator, comp, evaluate, stop_cond, sel_func, mate_func, mut_func);

	Individual v = g_alg.alg();

	std::cout << v << std::endl;

	return 0;
}
