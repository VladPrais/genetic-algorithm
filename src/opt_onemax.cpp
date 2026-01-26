#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "ga.h"

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist_pb(0.0, 1.0);
std::uniform_int_distribution<int> zero_one(0, 1);

typedef std::vector<double> GeneType;
typedef int FitnessType;
typedef ga::BaseIndividual<GeneType, FitnessType> Individual;

std::ostream& operator<<(std::ostream &stream, const Individual &individ)
{
	for(auto i = individ.genes.begin(); i != individ.genes.end(); i++)
	{
		stream << *i << ' ';
	}
	stream << individ.fitness;
	return stream;
}

const int len_genes = 3;
const int onemax_len_genes = 60;
const int onemax_max_gen = 100;
const int onemax_pop_size = 200;
const int onemax_elite = 0;

//
// Just onemax problem only
//

Individual onemax_generator(std::mt19937 &engine)
{
	GeneType genes(onemax_len_genes);
	std::generate(genes.begin(), genes.end(), [&engine](){ return zero_one(engine); });

	return Individual(genes);
}

bool onemax_comparator(const Individual &lhs, const Individual &rhs)
{
	return lhs.fitness > rhs.fitness;
}

FitnessType onemax_evaluate(Individual &individ)
{
	return std::accumulate(individ.genes.begin(), individ.genes.end(), 0);
}

bool onemax_stop_cond(Individual &individ)
{
	return std::accumulate(individ.genes.begin(), individ.genes.end(), 0) >= onemax_len_genes;
}

//
//
//

Individual generator(std::mt19937 &engine)
{
	GeneType genes(len_genes);
	std::generate(genes.begin(), genes.end(), [&engine](){ return dist_pb(engine); });

	return Individual(genes);
}

bool comparator(const Individual &lhs, const Individual &rhs)
{
	return lhs.fitness < rhs.fitness;
}

FitnessType evaluate(Individual &it)
{
	double cxpb = it.genes[0], mtpb = it.genes[1], mgpb = it.genes[2];

	if (cxpb < 0.0 || cxpb > 1.0)
	{
		cxpb = dist_pb(mt);
		//it.fitness += 10 * cxpb * cxpb;
	}
	if (mtpb < 0.0 || mtpb > 1.0)
	{
		mtpb = dist_pb(mt);
		//it.fitness += 10 * mtpb * mtpb;
	}
	if (mgpb < 0.0 || mgpb > 1.0)
	{
		mgpb = dist_pb(mt);
		//it.fitness += 10 * mgpb * mgpb;
	}

	ga::tournament<Individual> selection(3);
	ga::crossover<Individual> crossover(cxpb, ga::cx_one_point<Individual>());
	ga::mutation<Individual> mutation(mtpb, mgpb, ga::mut_inverse<Individual>());

	ga::SimpleGenetic<GeneType, FitnessType> g_alg(onemax_max_gen, onemax_pop_size, onemax_elite, onemax_generator, onemax_comparator, onemax_evaluate, onemax_stop_cond, selection, crossover, mutation);
	g_alg.output = false;
	g_alg();

	//std::cout << best << std::endl;

	return g_alg.iter_until_convergence;
}

bool stop_cond(Individual &individ)
{
	return individ.fitness <= 13;
}


int main(void)
{
	int max_gen = 100, pop_size = 200, elite = 0;
	double cxpb = 0.7, mtpb = 0.2, mgpb = 0.01;

	ga::tournament<Individual> selection(3);
	ga::crossover<Individual> crossover(cxpb, ga::cx_one_point<Individual>());
	ga::mutation<Individual> mutation(mtpb, mgpb, ga::mut_gaussian<Individual>(0.5, 0.1));

	//ga::AdvancedGenetic<GeneType, FitnessType> g_alg(max_gen, pop_size, elite, generator, comparator, evaluate, stop_cond, selection, crossover, mutation);
	ga::SimpleGenetic<GeneType, FitnessType> g_alg(max_gen, pop_size, elite, generator, comparator, evaluate, stop_cond, selection, crossover, mutation);
	g_alg.output = true;
	
	Individual best = g_alg();

	cxpb = best.genes[0];
	mtpb = best.genes[1];
       	mgpb = best.genes[2];

	for(auto v: best.genes)
	{
		std::cout << v << ' ';
	}
	std::cout << ' ' << best.fitness << std::endl;

	//ga::tournament<Individual> selection(3);
	//ga::crossover<Individual> crossover(cxpb, ga::cx_one_point<Individual>());
	//ga::mutation<Individual> onemax_mutation(mtpb, mgpb, ga::mut_inverse<Individual>());

	//ga::SimpleGenetic<GeneType, FitnessType> onemax(onemax_max_gen, onemax_pop_size, onemax_elite, onemax_generator, onemax_comparator, onemax_evaluate, onemax_stop_cond, selection, crossover, onemax_mutation);
	//onemax();

	return 0;
}
