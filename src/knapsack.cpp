#include "ga.h"

struct thing
{
	int weight;
	int utility;

	thing(void) = default;

	thing(int weight, int utility): weight(weight), utility(utility)
	{	}
};

const int _max_weight_thing = 10;
const int _max_utility_thing = 100;
const int _things_count = 50;
const int _max_weight = (_max_weight_thing * _things_count) / 2;
const int _max_utility = (_max_utility_thing * _things_count) / 2;

std::vector<thing> _things(_things_count);
std::uniform_int_distribution<int> dist(0, 1);

typedef std::vector<int> GeneType;
typedef int FitnessType;
typedef ga::BaseIndividual<GeneType, FitnessType> Individual;

Individual generator(std::mt19937 &engine)
{
	std::vector<int> genes(_things_count);

	std::generate(genes.begin(), genes.end(), [&engine](){ return dist(engine); });

	return Individual(genes);
}

bool comparator(const Individual &lhs, const Individual &rhs)
{
	return lhs.fitness > rhs.fitness;
}

FitnessType evaluate(Individual &it)
{
	int total_mass = std::inner_product(it.genes.begin(), it.genes.end(), _things.begin(), 0, std::plus<int>(), [](int flag, thing &th){ return flag * th.weight; });
	int total_utility = std::inner_product(it.genes.begin(), it.genes.end(), _things.begin(), 0, std::plus<int>(), [](int flag, thing &th){ return flag * th.utility; });
	int total_error = 0;

	if (total_mass > _max_weight)
	{
		double error = _max_weight - total_mass;
		total_error += error;
	}

	return total_utility - total_error;
//	return (total_utility > total_error ? total_utility - total_error : 0);
}

bool stop_cond(Individual &best)
{
	return best.fitness >= _max_utility;
}

std::ostream& operator<<(std::ostream &stream, const Individual &it)
{
	for(auto i = it.genes.begin(); i != it.genes.end(); i++)
	{
		stream << *i << ' ';
	}
	double total_mass = std::inner_product(it.genes.begin(), it.genes.end(), _things.begin(), 0.0, std::plus<double>(), [](int flag, thing &th){ return flag * th.weight; });
	stream << ' ' << it.fitness << "  " << total_mass;
	return stream;
}

int main(void)
{
	std::mt19937 mt(1);
	std::uniform_int_distribution<int> dist_w(1, _max_weight_thing);
	std::uniform_int_distribution<int> dist_u(0, _max_utility_thing);
	std::uniform_real_distribution<double> pb(0.0, 1.0);

	std::generate(_things.begin(), _things.end(), [&mt, &dist_w, &dist_u]()->thing{ return {dist_w(mt), dist_u(mt)}; });

	int max_generations = 300, pop_size = 3e3, elite = 0, tourn_size = 3;
	double cxpb = 0.6, mtpb = 0.2, mgpb = 0.01;
	
	ga::tournament<Individual> sel(tourn_size);
	ga::crossover<Individual> mate(cxpb, ga::cx_one_point<Individual>());
	ga::mutation<Individual> mut(mtpb, mgpb, ga::mut_inverse<Individual>());

	ga::BaseGenetic<GeneType, FitnessType> alg(generator, comparator, evaluate, stop_cond, sel, mate, mut, max_generations, pop_size, elite);

	ga::GeneticOutput ga_out = alg();

	Individual best = ga_out.best_ever;
	std::cout << best << std::endl;

	for(auto &th: _things)
	{
		std::cout << th.weight << ' ' << th.utility << std::endl;
	}
	int mw = std::accumulate(_things.begin(), _things.end(), 0, [](int x, thing &th){ return x + th.weight; });
	int mu = std::accumulate(_things.begin(), _things.end(), 0, [](int x, thing &th){ return x + th.utility; });
	std::cout << mw << ' ' << mu << std::endl;

	return 0;
}
