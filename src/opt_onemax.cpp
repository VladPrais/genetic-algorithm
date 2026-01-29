#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "ga.h"

const int LEN_GENES = 4;
const int ONEMAX_LEN_GENES = 100;
const int ONEMAX_MAX_GEN = 100;
const int ONEMAX_POP_SIZE = 300;

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist_pb(0.0, 1.0);
std::uniform_int_distribution<> dist_elite(0, ONEMAX_POP_SIZE - 1);
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

//
// Just onemax problem only
//

Individual onemax_generator(std::mt19937 &engine)
{
	GeneType genes(ONEMAX_LEN_GENES);
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
	return std::accumulate(individ.genes.begin(), individ.genes.end(), 0) >= ONEMAX_LEN_GENES;
}

//
//
//

Individual generator(std::mt19937 &engine)
{
	GeneType genes(LEN_GENES);
	std::generate(genes.begin(), genes.end() - 1, [&engine](){ return dist_pb(engine); });
	genes[LEN_GENES - 1] = dist_elite(engine);

	return Individual(genes);
}

bool comparator(const Individual &lhs, const Individual &rhs)
{
	return lhs.fitness < rhs.fitness;
}

FitnessType evaluate(Individual &it)
{
	double cxpb = it.genes[0], mtpb = it.genes[1], mgpb = it.genes[2], elite = it.genes[3];

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
	if (elite < 0 || elite > ONEMAX_POP_SIZE)
	{
		elite = dist_elite(mt);
	}

	ga::tournament<Individual> selection(3);
	ga::crossover<Individual> crossover(cxpb, ga::cx_one_point<Individual>());
	ga::mutation<Individual> mutation(mtpb, mgpb, ga::mut_inverse<Individual>());

	ga::BaseGenetic<GeneType, FitnessType> g_alg(onemax_generator, onemax_comparator, onemax_evaluate, onemax_stop_cond, selection, crossover, mutation, ONEMAX_MAX_GEN, ONEMAX_POP_SIZE, elite);
	g_alg.output = false;
	ga::GeneticOutput gen_out = g_alg();

	return gen_out.generation_count;
}

bool stop_cond(Individual &individ)
{
	return individ.fitness <= 15;
}

int main(void)
{
	int max_gen = 20, pop_size = 200, elite = 10;
	double cxpb = 0.7, mtpb = 0.2, mgpb = 0.01;

	ga::tournament<Individual> selection(3);
	ga::crossover<Individual> crossover(cxpb, ga::cx_one_point<Individual>());
	ga::mutation<Individual> mutation(mtpb, mgpb, ga::mut_gaussian<Individual>(0.5, 0.1));

	//ga::AdvancedGenetic<GeneType, FitnessType> g_alg(generator, comparator, evaluate, stop_cond, selection, crossover, mutation, max_gen, pop_size, elite);
	ga::BaseGenetic<GeneType, FitnessType> g_alg(generator, comparator, evaluate, stop_cond, selection, crossover, mutation, max_gen, pop_size, elite);
	g_alg.output = true;
	
	Individual best = g_alg().best_ever;
	std::cout << best << std::endl;

	cxpb = best.genes[0];
	mtpb = best.genes[1];
       	mgpb = best.genes[2];
	elite = best.genes[3];

	ga::tournament<Individual> selection_2(3);
	ga::crossover<Individual> crossover_2(cxpb, ga::cx_one_point<Individual>());
	ga::mutation<Individual> mutation_2(mtpb, mgpb, ga::mut_inverse<Individual>());

	ga::BaseGenetic<GeneType, FitnessType> onemax(onemax_generator, onemax_comparator, onemax_evaluate, onemax_stop_cond, selection_2, crossover_2, mutation_2, ONEMAX_MAX_GEN, ONEMAX_POP_SIZE, elite);
	onemax.output = true;

	best = onemax().best_ever;
	std::cout << best << std::endl;

	return 0;
}
