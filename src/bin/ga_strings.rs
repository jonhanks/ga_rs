use std::sync::Arc;
use rand;
use rand::Rng;
use ga;
use clap::Parser;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    #[arg(long, default_value="helloworld")]
    phrase: String,
    #[arg(long, default_value_t=1000)]
    max_generations: usize,
    #[arg(long, default_value_t=1000)]
    population_size: usize,
    #[arg(short, long, default_value_t=false)]
    verbose: bool,
}

#[derive(Clone)]
struct StringIndividual {
    genes: [u8; 10]
}

impl StringIndividual {
    fn new() -> Self {
        let mut r = rand::thread_rng();
        StringIndividual{genes: [
            'a' as u8 + r.gen_range(0..26),
            'a' as u8 + r.gen_range(0..26),
            'a' as u8 + r.gen_range(0..26),
            'a' as u8 + r.gen_range(0..26),
            'a' as u8 + r.gen_range(0..26),
            'a' as u8 + r.gen_range(0..26),
            'a' as u8 + r.gen_range(0..26),
            'a' as u8 + r.gen_range(0..26),
            'a' as u8 + r.gen_range(0..26),
            'a' as u8 + r.gen_range(0..26),
        ]}
    }
}

impl ga::Individual for StringIndividual {
    fn mutate(&self) -> Self {
        let mut r = rand::thread_rng();
        let i = r.gen_range(0..self.genes.len());
        let mut ind = self.clone();
        ind.genes[i] = (r.gen_range(0..26) + 'a' as u8) as u8;
        ind
    }
}

struct StringGenerator {}

impl ga::Generator<StringIndividual> for StringGenerator {
    fn generate(&self) -> StringIndividual {
        StringIndividual::new()
    }

    fn evolve(&self, a: &StringIndividual, b: &StringIndividual) -> StringIndividual {
        let mut ind: StringIndividual = a.clone();
        let mut r = rand::thread_rng();
        let mid = r.gen_range(0..9);
        for i in mid..10 {
            ind.genes[i] = b.genes[i];
        }
        ind
    }
}

fn string_fitness(phrase: String) -> Arc<dyn Fn(&StringIndividual) -> f32 + Send + Sync> {
    let bytes = phrase.as_bytes();
    if bytes.len() != 10 {
        panic!("The phrase must be 10 letters long, and only a-z")
    }
    let mut data = Vec::new();
    data.reserve(bytes.len());
    for i in 0..10 {
        if bytes[i] < 'a' as u8 || bytes[i] > 'z' as u8 {
            panic!("The phrase must be 10 letters long, and only a-z")
        }
        data.push(bytes[i])
    }
    Arc::new(move |s: &StringIndividual| -> f32 {
        let mut score = 0.0;
        for i in 0..10 {
            let delta = data[i as usize] as i32 - s.genes[i as usize] as i32;
            score += 1.0 - delta.abs() as f32;
        }
        score
    })
}

fn main() {
    let args = Args::parse();

    let gen = StringGenerator{};
    let fitness = string_fitness(args.phrase.clone());
    let mut pop = ga::Population::new(args.population_size, &gen, fitness.clone());

    let mut generations = 1;

    for g in 1..args.max_generations {
        if args.verbose {
            println!("{0})", g);
            pop.individuals.iter().take(5).for_each(|ind| {
                println!("\t{0} {1}", std::str::from_utf8(&ind.individual.genes).unwrap(), ind.fitness)
            });
        }
        if pop.individuals.first().unwrap().fitness == 10.0 {
            break;
        }
        pop = pop.evolve(&gen,fitness.clone());
        generations += 1;
    }

    println!("After {0} Generations:", generations);
    pop.individuals.iter().take(5).for_each(|ind| {
        println!("\t{0} {1}", std::str::from_utf8(&ind.individual.genes).unwrap(), ind.fitness)
    });
}
