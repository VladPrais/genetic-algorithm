#include "ga.h"

typedef std::vector<double> GeneType;
typedef double FitnessType;
typedef ga::BaseIndividual<GeneType, FitnessType> Individual;

const int dim = 100;
const double pi = 3.1415;
const double bound = 5.12;
std::uniform_real_distribution<double> dist(-bound, bound);

Individual generator(std::mt19937 &engine)
{
	std::vector<double> genes(dim);
	std::generate(genes.begin(), genes.end(), [&engine](){ return dist(engine); });

	return Individual(genes);
}

bool comparator(const Individual &lhs, const Individual &rhs)
{
	return lhs.fitness < rhs.fitness;
}

FitnessType evaluate(Individual &it)
{
	double A = 10.0;
	double s = A * dim;

	for(int i = 0; i < dim; i++)
	{
		double x = it.genes[i];
		s += x * x - A * std::cos(2 * pi * x);
	}
	return s;
}

bool stop_cond(Individual &best)
{
	return std::abs(best.fitness) < 1e-8;
}

std::ostream& operator<<(std::ostream &stream, const Individual &it)
{
	for(auto i = it.genes.begin(); i != it.genes.end(); i++)
	{
		stream << *i;
		if (i + 1 != it.genes.end())
			stream << ' ';
	}
	stream << ' ' << it.fitness;
	return stream;
}

int main(void)
{
	int max_generations = 200, pop_size = 1e5, elite = 1e2, tourn_size = 4;
	double cxpb = 0.4, mtpb = 0.2, mgpb = 0.01;

	ga::tournament<Individual> sel(tourn_size);
	ga::crossover<Individual> mate(cxpb, ga::cx_one_point<Individual>());
	ga::mutation<Individual> mut(mtpb, mgpb, ga::mut_gaussian<Individual>(0.0, 1.0));

	ga::ParallelGenetic<GeneType, FitnessType> alg(generator, comparator, evaluate, stop_cond, sel, mate, mut, max_generations, pop_size, elite);

	ga::GeneticOutput ga_out = alg();
	Individual best = ga_out.best_ever;

	std::cout << best << std::endl;

	return 0;
}
