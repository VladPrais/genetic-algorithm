#ifndef __GA_H__
#define __GA_H__

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

template <typename GeneType, typename FitnessType>
class BaseIndividual
{
	public:

	GeneType genes;
	FitnessType fitness;

	BaseIndividual() = default;

	BaseIndividual(GeneType genes):
		genes(genes)
	{	}

};

template <typename GeneType, typename FitnessType>
class BaseGeneration
{
	typedef BaseIndividual<GeneType, FitnessType> Individual;
	typedef std::vector<Individual> Population;
	typedef std::function<Individual(void)> Generator;
	typedef std::function<bool(Individual&, Individual&)> Comparator;
	typedef std::function<FitnessType(Individual&)> Fitness;

	Population population;
	Generator generator;
	Comparator comp;
	Fitness fitnes;

	int pop_size;

	void init(void)
	{
		std::generate(population.begin(), population.end(), generator);
	}

	void compute_fitness(void)
	{
		for(Individual &i: population)
		{
			i.fitness = fitness(i);
		}
	}

	public:

	BaseGeneration(Generator generator, int pop_size, Comparator comp, Fitness fitnes):
		generator(generator),
		pop_size(pop_size),
		population(pop_size),
		comp(comp),
		fitnes(fitnes)
	{
		init();
		compute_fitness();
	}

	int get_pop_size(void)
	{
		return this->pop_size;
	}

	Individual get_best(void)
	{
		auto iter_max = std::max_element(population.begin(), population.end(), comp);
		int n = std::distance(population.begin(), iter_max);

		Individual the_best = population[n];

		return the_best;
	}

	Individual get_worst(void)
	{
		auto iter_min = std::min_element(population.begin(), population.end(), comp);
		int n = std::distance(population.begin(), iter_min);

		Individual the_worst = population[n];

		return the_worst;
	}
	
	void print(Individual &individual)
	{
		for(auto g: individual.genes)
		{
			std::cout << g << ' ';
		}
		std::cout << '\t' << individual.fitness << std::endl;
	}

	void print()
	{
		for(Individual i: population)
		{
			print(i);
		}
	}
};

#endif
