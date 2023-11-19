use rayon::prelude::*;
use std::sync::Arc;
use rand::Rng;

pub trait Individual: Clone {
    fn mutate(&self) -> Self;
}

pub trait Generator<I> {
    fn generate(&self) -> I;
    fn evolve(&self, a: &I, b: &I) -> I;
}

#[derive(Clone)]
pub struct GradedIndividual<I: Individual>
where
    I: Send + Sync
{
    pub individual: I,
    pub fitness: f32,
}

impl<I> GradedIndividual<I>
where
    I: Individual + Send + Sync
{
    fn new(individual: I, fitness: f32) -> Self
    where
    {
        Self{
            individual,
            fitness,
        }
    }
}

pub struct Population<I>
where
    I: Individual + Send + Sync
{
    pub individuals: Vec<GradedIndividual<I>>,
}

impl<I> Population<I>
where
    I: Individual + Send + Sync
{
    pub fn new<G>(size: usize, generator: &G, fitness: Arc<dyn Fn(&I) -> f32 + Send + Sync>) -> Self
    where
        G: Generator<I> + Send + Sync,
    {
        let mut population : Vec<GradedIndividual<I>> = Vec::new();
        population.reserve(size);
        (1..size).into_par_iter().map(|_| -> GradedIndividual<I> {
            let individual = generator.generate();
            let score = fitness(&individual);
            GradedIndividual::new(individual, score)
        }).collect_into_vec(&mut population);

        population.par_sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        Population{
            individuals: population
        }
    }

    pub fn evolve<G>(&self, generator: &G, fitness: Arc<dyn Fn(&I) -> f32 + Send + Sync>) -> Self
    where
        G: Generator<I> + Send + Sync
    {
        let mut population : Vec<GradedIndividual<I>> = Vec::new();
        population.reserve(self.individuals.len());

        // copy the first 10% over
        let copy_count = (self.individuals.len() as f32 * 0.1) as usize;

        let copy_it = self.individuals.par_iter().take(copy_count).map(|entry|-> GradedIndividual<I> {
            (*entry).clone()
        });

        let mutate_it = self.individuals[copy_count..].par_iter().step_by(2).map(|entry| -> GradedIndividual<I> {
            let ind = entry.individual.mutate();
            let score = fitness(&entry.individual);
            GradedIndividual::<I>{
                individual: ind,
                fitness: score,
            }
        });
        let evolve_it = self.individuals[copy_count+1..].par_iter().step_by(2).map(|entry| -> GradedIndividual<I> {
            let mut r = rand::thread_rng();
            let other_index = r.gen_range(0..self.individuals.len());
            let ind = generator.evolve(&entry.individual, &self.individuals[other_index].individual);
            let score = fitness(&ind);
            GradedIndividual::<I>{
                individual: ind,
                fitness: score,
            }
        });

        copy_it.interleave(mutate_it).interleave(evolve_it).collect_into_vec(&mut population);

        population.par_sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        Population{
            individuals: population
        }
    }
}