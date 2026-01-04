#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

std::random_device rd;
std::mt19937 engine(rd());
std::uniform_real_distribution PB(0.0, 1.0);

class Individual
{
	public:

	std::vector<int> genes;
	int fitness;

	Individual() = default;

	Individual(std::vector<int> genes): genes(genes)
	{	}

	bool operator<(const Individual &I) const
	{
		return this->fitness < I.fitness;
	}
};

std::ostream& operator<<(std::ostream &stream, Individual &I)
{
	for(auto g: I.genes)
	{
		stream << g << ' ';
	}
	stream << " (" << I.fitness << ')';

	return stream;
}

std::vector<Individual> selection(std::vector<Individual> &population, int tourn_size)
{
	int pop_size = population.size();
	std::vector<Individual> aspirants;

	std::uniform_int_distribution dist(0, pop_size - 1);

	for(int i = 0; i < pop_size; i++)
	{
		std::vector<Individual> temp;
		
		for(int j = 0; j < tourn_size; j++)
		{
			int n = dist(engine);
			temp.push_back(population[n]);
		}

		auto best = std::max_element(temp.begin(), temp.end());
		int ind_best = std::distance(temp.begin(), best);

		aspirants.push_back(temp[ind_best]);
	}

	return aspirants;
}

void cx_oper(Individual &I1, Individual &I2)
{
	int len_gen = I1.genes.size();
	std::uniform_int_distribution dist(0, len_gen - 1);

	int dot = dist(engine);

	for(int i = 0; i < dot; i++)
	{
		std::swap(I1.genes[i], I2.genes[i]);
	}
}

void crossover(std::vector<Individual> &aspirants, double CXPB)
{
	int pop_size = aspirants.size();

	for(int i = 0; i < pop_size; i += 2){
		
		double pb = PB(engine);

		if(pb < CXPB)
		{
			cx_oper(aspirants[i], aspirants[i + 1]);
		}
	}
}

void mt_oper(Individual &I)
{
	int len_gen = I.genes.size();
	std::uniform_int_distribution dist(0, len_gen - 1);

	int dot = dist(engine);

	I.genes[dot] = !I.genes[dot];
}

void mutation(std::vector<Individual> &child, double MTPB)
{
	int pop_size = child.size();

	for(int i = 0; i < pop_size; i++)
	{
		double pb = PB(engine);

		if(pb < MTPB)
		{
			mt_oper(child[i]);
		}
	}
}

Individual init(int len_genes)
{
	std::vector<int> genes;

	std::uniform_int_distribution dist(0, 1);

	for(int i = 0; i < len_genes; i++)
	{
		genes.push_back(dist(engine));
	}

	Individual I(genes);

	return I;
}

double fitness(Individual &I)
{
	return std::accumulate(I.genes.begin(), I.genes.end(), 0);
}

void fitness_pop(std::vector<Individual> &population)
{
	for(auto &i: population)
	{
		i.fitness = fitness(i);
	}
}

std::vector<double> analysis(std::vector<Individual> &population)
{
	auto min_it = std::min_element(population.begin(), population.end());
	auto max_it = std::max_element(population.begin(), population.end());

	int ind_min = std::distance(population.begin(), min_it);
	int ind_max = std::distance(population.begin(), max_it);

	double min = population[ind_min].fitness;
	double max = population[ind_max].fitness;

	double avg = 0.0;
	for(auto i: population)
	{
		avg += i.fitness;
	}
	avg /= population.size();

	std::vector<double> results{min, max, avg};

	return results;
}

std::vector<std::vector<double>> ga(int pop_size, int len_genes, int generations, int tourn_size, double CXPB, double MTPB)
{
	std::vector<Individual> population(pop_size);

	auto _init = std::bind(init, len_genes);
	std::generate(population.begin(), population.end(), _init);
	fitness_pop(population);

	std::cout << "generation  min  max  avg" << std::endl;

	std::vector<double> results = analysis(population);
	char space = '\t';

	std::vector<double> MIN, MAX, AVG;

	std::cout << 0 << space << results[0] << space << results[1] << space << results[2] << std::endl;

	MIN.push_back(results[0]);
	MAX.push_back(results[1]);
	AVG.push_back(results[2]);

	for(int i = 1; i < generations + 1; i++)
	{
		population = selection(population, tourn_size);
		crossover(population, CXPB);
		mutation(population, MTPB);
		fitness_pop(population);

		results = analysis(population);

		std::cout << i << space << results[0] << space << results[1] << space << results[2] << std::endl;

		MIN.push_back(results[0]);
		MAX.push_back(results[1]);
		AVG.push_back(results[2]);

		if(results[1] == len_genes)
		{
			break;
		}
	}

	std::vector<std::vector<double>> RESULTS{MIN, MAX, AVG};

	return RESULTS;
}

int main(void)
{

	Individual i1 = init(8), i2 = init(8);
	i1.fitness = fitness(i1);
	i2.fitness = fitness(i2);

	std::cout << i1 << std::endl << i2 << std::endl;

	cx_oper(i1, i2);
	i1.fitness = fitness(i1);
	i2.fitness = fitness(i2);

	std::cout << i1 << std::endl << i2 << std::endl;

	mt_oper(i1);
	i1.fitness = fitness(i1);
	std::cout << i1 << std::endl;
	/*
	 * standart
	 */

	/*
	int pop_size = 1000, len_genes = 300, generations = 10000, tourn_size = 3;
	double CXPB = 0.7, MTPB = 0.1;

	std::vector<std::vector<double>> results;

	results = ga(pop_size, len_genes, generations, tourn_size, CXPB, MTPB);
	int size = results[0].size();

	*/
	/*
	std::ofstream file("ga1.txt");
	for(int i = 0; i < size; i++)
	{
		file << results[0][i] << ' ' << results[1][i] << ' ' << results[2][i] << std::endl;
	}
	file.close();

	*/
	/*
	 * increse tourn_size
	 */
/*
	tourn_size = 9;

	results = ga(pop_size, len_genes, generations, tourn_size, CXPB, MTPB);
	size = results[0].size();

	std::ofstream file2("ga2.txt");
	for(int i = 0; i < size; i++)
	{
		file2 << results[0][i] << ' ' << results[1][i] << ' ' << results[2][i] << std::endl;
	}
	file2.close();
*/
	/*
	 * increase MTPB
	 */

	/*
	tourn_size = 3;
	MTPB = 0.4;
	results = ga(pop_size, len_genes, generations, tourn_size, CXPB, MTPB);
	size = results[0].size();

	std::ofstream file3("ga3.txt");
	for(int i = 0; i < size; i++)
	{
		file3 << results[0][i] << ' ' << results[1][i] << ' ' << results[2][i] << std::endl;
	}
	file3.close();

	*/
	/*
	 * increase CXPB
	 */

	/*
	MTPB = 0.1;
	CXPB = 0.9;

	results = ga(pop_size, len_genes, generations, tourn_size, CXPB, MTPB);
	size = results[0].size();

	std::ofstream file4("ga4.txt");
	for(int i = 0; i < size; i++)
	{
		file4 << results[0][i] << ' ' << results[1][i] << ' ' << results[2][i] << std::endl;
	}
	file4.close();

	*/

	return 0;
}
