#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "ga.h"

std::random_device rd;
std::mt19937 engine(rd());
std::uniform_int_distribution dist(0, 1);

typedef std::vector<int> GeneType;
typedef int FitnessType;
typedef ga::BaseIndividual<GeneType, FitnessType> Individual;

const size_t LEN_GENES = 20;

Individual generator(void)
{
	GeneType v(LEN_GENES);

	std::generate(v.begin(), v.end(), [](){ return dist(engine); });

	Individual I(v);

	return I;
}

bool comp(Individual &i1, Individual &i2)
{
	return i1.fitness < i2.fitness;
}

FitnessType evaluate(Individual &i)
{
	int s = 0;

	for(auto j: i.genes)
	{
		s += j;
	}

	return s;
}

bool stop_cond(Individual& i)
{
	if(i.fitness == LEN_GENES)
	{
		return true;
	}
	return false;
}

Individual sel_func(std::vector<Individual> &population)
{
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

void mut_func(Individual &i)
{
	int len_genes = i.genes.size();

	std::uniform_int_distribution dot(0, len_genes - 1);

	int p = dot(engine);

	i.genes[p] = !i.genes[p];
}

int main(void)
{
	int pop_size = 100, max_generations = 50, elite = 2;
	double cxpb = 0.7, mtpb = 0.1;

	ga::BaseGeneration<GeneType, FitnessType> base_gen(pop_size, generator, comp, evaluate);

	Individual i = sel_func(base_gen.get());

	ga::GeneticAlgorithm<GeneType, FitnessType> g_alg(max_generations, pop_size, cxpb, mtpb, elite, generator, comp, evaluate, stop_cond, sel_func, mate_func, mut_func);

	g_alg.alg();

	return 0;
}
