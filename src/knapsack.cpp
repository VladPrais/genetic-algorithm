#include "ga.h"

struct thing
{
	int weight;
	int utility;

	thing(void) = default;

	thing(int weight, int utility): weight(weight), utility(utility)
	{	}
};

enum Consts
{
	MAX_WEIGHT = 50,
	MAX_THINGS = 70
};

std::uniform_int_distribution<int> dist_w(0, MAX_WEIGHT), dist_u(0, 50);

typedef std::vector<thing> GeneType;
typedef int FitnessType;
typedef ga::BaseIndividual<GeneType, FitnessType> Individual;

Individual generator(std::mt19937 &engine)
{
	std::vector<thing> genes(MAX_WEIGHT);

	std::generate(genes.begin(), genes.end(), [&engine]()->thing{ return {dist_w(engine), dist_u(engine)}; });

	return Individual(genes);
}

bool comparator(Individual &lhs, Individual &rhs)
{
	return lhs.fitness > rhs.fitness;
}

FitnessType evaluate(Individual &it)
{

	return 0;
}

int main(void)
{
	std::mt19937 mt(1);
	Individual i1 = generator(mt);

	for(auto th: i1.genes)
	{
		std::cout << th.weight, 
	}

	return 0;
}
