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
std::mt19937 mt(rd());
std::uniform_int_distribution dist(0, 1);

const size_t LEN_GENES = 1e1;

// The function generates the one Individual. Will be used in Genetic Algorithm to create initial population.
Individual generate(void)
{
	GeneType v(LEN_GENES);
	std::generate(v.begin(), v.end(), [](){ return dist(mt); });
	return Individual(v);
}

// The function to get fitness value.
FitnessType evaluate(Individual &i)
{
	return std::accumulate(i.get_genes().begin(), i.get_genes().end(), 0);
}

// Stop conditions. True if stop else False.
bool stop_cond(Individual& i)
{
	if(i.get_fitness() >= LEN_GENES)
		return true;
	return false;
}

std::ostream& operator<<(std::ostream &stream, const Individual &i)
{
	for(auto g: i.get_genes())
	{
		stream << g << ' ';
	}
	stream << "  " << i.get_fitness();

	return stream;
}

bool compare(const Individual &lhs, const Individual &rhs)
{
	return lhs.get_fitness() > rhs.get_fitness();
}

int main(void)
{
	/*
	int pop_size = 1e3, gen_limit = 5e3, elite = 2;
	double cxpb = 0.6, mtpb = 0.2, mgpb = 0.01;
	int tourn_size = 3;

	std::mt19937 engine(std::random_device{}());
	auto sel = [tourn_size, &engine](auto b1, auto e1, auto b2, auto e2){ sel_tournament(b1, e1, b2, e2, compare, engine, tourn_size); };
	auto cross = [&engine](auto lhs, auto rhs){ cross_one_point(lhs, rhs, engine); };
	auto mut = [&engine](auto i){ mut_bit_not(i, engine); };

	auto g_alg = ga::make_common<Individual>(generate, compare, evaluate, stop_cond);
	Individual best = g_alg(gen_limit, pop_size, elite, cxpb, mtpb, sel, cross, mut);

	std::cout << best << std::endl;
	*/

	Individual i = generate();
	i.set_fitness(evaluate(i));
	std::cout << i << std::endl;

	ga::mut_bit_not<Individual>(i, mt);
	i.set_fitness(evaluate(i));
	std::cout << i << std::endl;

	Individual i2 = generate();
	i2.set_fitness(evaluate(i2));
	std::cout << i2 << std::endl;

	std::cout << "---" << std::endl;

	cross_one_point(i, i2, mt);
	std::cout << i << std::endl << i2 << std::endl;

	return 0;
}
