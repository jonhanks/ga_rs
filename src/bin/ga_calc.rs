use ga::Individual;
use std::fmt::{Display, Formatter};
use std::num::Wrapping;
use std::sync::Arc;
use rand;
use rand::Rng;
use ga;
use clap::Parser;
use rayon::iter::split;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    #[arg(long, default_value_t=20000)]
    max_generations: usize,
    #[arg(long, default_value_t=10000)]
    population_size: usize,
    #[arg(short, long, default_value_t=false)]
    verbose: bool,
}

struct Stats {
    instructions_issued: i32,
    invalid_instructions: i32,
}

impl Stats {
    fn new() -> Self {
        Stats{
            instructions_issued: 0,
            invalid_instructions: 0,
        }
    }
}

enum ExitType {
    Timeout,
    Abort,
}

struct SVM {
    memory: Vec<i32>,
    stack: Vec<i32>,
    stats: Stats,
}

impl SVM {
    fn new(words: usize, stack_size: usize) -> Self {
        let mut vm = SVM {
            memory: Vec::new(),
            stack: Vec::new(),
            stats: Stats::new(),
        };
        vm.memory.resize(words, 0);
        vm.stack.reserve(stack_size);
        vm
    }

    fn peek_mem(&self, address: usize) -> i32 {
        if let Some(v) = self.memory.get(address) {
            return *v;
        }
        0
    }

    fn poke_mem(& mut self, address: usize, value: i32) {
        if address < self.memory.len() {
            self.memory[address] = value;
        }
    }

    fn pop_stack(& mut self) -> i32 {
        if self.stack.is_empty() {
            return 0;
        }
        let val = self.stack.pop().unwrap();
        val
    }

    fn push_stack(&mut self, val: i32) {
        if self.stack.len() < self.stack.capacity() {
            self.stack.push(val);
        }
    }

    fn reset_state(&mut self) {
        self.memory.fill(0);
        self.stack.clear();
        self.stats = Stats::new();
    }

    fn execute(&mut self, program: &Vec<OpCode>, max_steps: i32) -> ExitType {
        let mut ip: i32 = 0;
        let mut done = false;

        let mut bound_ip = |old_ip: i32| -> i32 {
            if old_ip < 0 {
                return 0;
            }
            if old_ip as usize >= program.len() {
                return 0;
            }
            old_ip
        };

        let mut remaining_steps = max_steps;
        let mut exit_type = ExitType::Timeout;

        while !done && remaining_steps > 0 {
            remaining_steps -= 1;
            ip = bound_ip(ip);

            let cur_op = &program[ip as usize];
            ip += 1;
            self.stats.instructions_issued += 1;

            match cur_op.code {
                Instruction::Nop => {},
                Instruction::BitOr => {
                    let a = self.pop_stack();
                    let b = self.pop_stack();
                    self.push_stack(a | b);
                },
                Instruction::BitAnd => {
                    let a = self.pop_stack();
                    let b = self.pop_stack();
                    self.push_stack(a & b);
                },
                Instruction::BitXor => {
                    let a = self.pop_stack();
                    let b = self.pop_stack();
                    self.push_stack(a ^ b);
                },
                Instruction::Add => {
                    let a = Wrapping(self.pop_stack());
                    let b = Wrapping(self.pop_stack());
                    let Wrapping(c) = a + b;
                    self.push_stack(c);
                },
                Instruction::Sub => {
                    let a = Wrapping(self.pop_stack());
                    let b = Wrapping(self.pop_stack());
                    let Wrapping(c) = a - b;
                    self.push_stack(c);
                },
                Instruction::Mult => {
                    let a = Wrapping(self.pop_stack());
                    let b = Wrapping(self.pop_stack());
                    let Wrapping(c) = a * b;
                    self.push_stack(c);
                },
                Instruction::Div => {
                    let dividend = Wrapping(self.pop_stack());
                    let divisor = Wrapping(self.pop_stack());
                    if divisor != Wrapping(0) {
                        let Wrapping(c) = dividend / divisor;
                        self.push_stack( c );
                    } else {
                        self.push_stack( i32::MAX);
                    }
                },
                Instruction::Push => { self.push_stack(cur_op.literal);},
                Instruction::Pop => { self.pop_stack(); },
                Instruction::PushDuplicate => {
                    let val = self.pop_stack();
                    self.push_stack(val);
                    self.push_stack(val);
                },
                Instruction::PushMem => {self.push_stack(self.peek_mem(cur_op.literal as usize));},
                Instruction::PopMem => {
                    let value = self.pop_stack();
                    self.poke_mem(cur_op.literal as usize, value);
                },
                Instruction::JumpRel => { ip += cur_op.literal; },
                Instruction::JumpEq => {
                    if self.pop_stack() == 0 {
                        ip += cur_op.literal;
                    }
                },
                Instruction::JumpGt => {
                    if self.pop_stack() > 0 {
                        ip += cur_op.literal;
                    }
                },
                Instruction::JumpLt => {
                    if self.pop_stack() < 0 {
                        ip += cur_op.literal;
                    }
                },
                Instruction::Abort => {
                    done = true;
                    exit_type = ExitType::Abort;
                }
            }
        };
        exit_type
    }
}

#[derive(Copy,Clone,Debug)]
enum Instruction {
    Nop,
    BitOr,
    BitAnd,
    BitXor,
    Add,
    Sub,
    Mult,
    Div,
    Push,
    Pop,
    PushDuplicate,
    PushMem,
    PopMem,
    JumpRel,
    JumpEq,
    JumpGt,
    JumpLt,
    Abort,
}

impl TryFrom<u8> for Instruction {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Instruction::Nop),
            1 => Ok(Instruction::BitOr),
            2 => Ok(Instruction::BitAnd),
            3 => Ok(Instruction::BitXor),
            4 => Ok(Instruction::Add),
            5 => Ok(Instruction::Sub),
            6 => Ok(Instruction::Mult),
            7 => Ok(Instruction::Div),
            8 => Ok(Instruction::Push),
            9 => Ok(Instruction::Pop),
            10 => Ok(Instruction::PushDuplicate),
            11 => Ok(Instruction::PushMem),
            12 => Ok(Instruction::PopMem),
            13 => Ok(Instruction::JumpRel),
            14 => Ok(Instruction::JumpEq),
            15 => Ok(Instruction::JumpGt),
            16 => Ok(Instruction::JumpLt),
            17 => Ok(Instruction::Abort),
            _ => Err("Value out of range for an instruction")
        }
    }
}

impl Into<u8> for Instruction {
    fn into(self) -> u8 {
        match self {
            Instruction::Nop => 0 ,
            Instruction::BitOr => 1 ,
            Instruction::BitAnd => 2 ,
            Instruction::BitXor => 3 ,
            Instruction::Add => 4 ,
            Instruction::Sub => 5 ,
            Instruction::Mult => 6 ,
            Instruction::Div => 7 ,
            Instruction::Push => 8 ,
            Instruction::Pop => 9 ,
            Instruction::PushDuplicate => 10,
            Instruction::PushMem => 11,
            Instruction::PopMem => 12,
            Instruction::JumpRel => 13,
            Instruction::JumpEq => 14,
            Instruction::JumpGt => 15,
            Instruction::JumpLt => 16,
            Instruction::Abort => 17,
        }
    }
}

#[derive(Clone,Debug)]
struct OpCode {
    code: Instruction,
    literal: i32,
}

impl OpCode {
    fn rand() -> Self {
        let mut r = rand::thread_rng();
        let a = Instruction::Abort;
        let end: u8 = a.try_into().unwrap();
        OpCode {
            code: Instruction::try_from(r.gen_range(0..end+1u8)).unwrap(),
            literal: r.gen_range(0..5),
        }
    }
}

impl Display for OpCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.code {
            Instruction::Nop => write!(f, "nop"),
            Instruction::BitOr => write!(f, "bit_or"),
            Instruction::BitAnd => write!(f, "bit_and"),
            Instruction::BitXor => write!(f, "bit_xor"),
            Instruction::Add => write!(f, "add"),
            Instruction::Sub => write!(f, "sub"),
            Instruction::Mult => write!(f, "mult"),
            Instruction::Div => write!(f, "div"),
            Instruction::Push => write!(f, "push {0}", self.literal),
            Instruction::Pop => write!(f, "pop"),
            Instruction::PushDuplicate => write!(f, "push_dup"),
            Instruction::PushMem => write!(f, "push ({0})", self.literal),
            Instruction::PopMem => write!(f, "pop_to {0}", self.literal),
            Instruction::JumpRel => write!(f, "jmp {0}", self.literal),
            Instruction::JumpEq => write!(f, "jmp_eq {0}", self.literal),
            Instruction::JumpGt => write!(f, "jmp_lt {0}", self.literal),
            Instruction::JumpLt => write!(f, "jmp_gt {0}", self.literal),
            Instruction::Abort => write!(f, "abort"),
        }
    }
}

#[derive(Clone)]
struct CalcIndividual {
    ops: Vec<OpCode>,
}

impl CalcIndividual {
    fn new() -> Self {
        let mut r = rand::thread_rng();
        let count = r.gen_range(5..25);
        let mut ops = Vec::new();
        ops.reserve(count);
        for _ in 0..count {
            ops.push(OpCode::rand());
        }
        CalcIndividual {
            ops,
        }
    }
}

impl ga::Individual for CalcIndividual {
    fn mutate(&self) -> Self {
        let mut r = rand::thread_rng();
        let mut action: i32 = r.gen_range(0..3);
        let l = self.ops.len();
        if l <= 2 && action == 1 {
            action = 2;
        }
        let action = action;
        let mut ops: Vec<OpCode> = Vec::new();
        match action {
            0 => {
                ops = self.ops.clone();
                ops[r.gen_range(0..l) as usize] = OpCode::rand();
            },
            1 => {
                let split_point = r.gen_range(0..l) as usize;
                ops.reserve(l+1);
                let mut i: usize = 0;
                self.ops.iter().filter_map(|op| -> Option<OpCode> {
                    let result = if i == split_point {
                        None
                    } else {
                        Some((*op).clone())
                    };
                    i += 1;
                    result
                }).for_each(|op| {
                    ops.push(op);
                })
            },
            _ => {
                ops.reserve(l+1);
                self.ops.iter().for_each(|op| {
                    ops.push((*op).clone());
                });
                ops.push(OpCode::rand());
            },
        }
        CalcIndividual{
            ops
        }
    }
}

struct Generator {}

impl ga::Generator<CalcIndividual> for Generator {
    fn generate(&self) -> CalcIndividual {
        CalcIndividual::new()
    }

    fn evolve(&self, a: &CalcIndividual, _: &CalcIndividual) -> CalcIndividual {
        (*a).mutate()
    }
}

fn get_val() -> i32 {
    rand::thread_rng().gen_range(1..10000)
}

fn fitness_function() -> Arc<dyn Fn(&CalcIndividual) -> f32 + Send + Sync> {
    Arc::new(move |subject: &CalcIndividual| -> f32 {
        let a = get_val();
        let b = get_val();
        //let c = get_val();
        let expected = ((a * b) + a) as f32;

        let mut vm = SVM::new(100, 100);
        vm.poke_mem(0, a);
        vm.poke_mem(1, b);
        //vm.poke_mem(2, c);
        let exit_type = vm.execute(&subject.ops, 25);
        let val = vm.peek_mem(3) as f32;
        let modifier: f32 = match exit_type {
            ExitType::Abort => {
                if expected == val {
                    let l = subject.ops.len();
                    if l < 10 {
                        2.0
                    } else if l < 15 {
                        1.8
                    } else if l < 20 {
                        1.7
                    } else if l < 25 {
                        1.6
                    } else if l < 30 {
                        1.5
                    } else {
                        1.0
                    }
                } else {
                    0.0
                }
            },
            ExitType::Timeout => {
                0.0
            },
        };
        modifier - ((expected-val) as f32).abs()
    })
}


fn main() {
    let args = Args::parse();

    let gen = Generator{};
    let fitness = fitness_function();
    let mut pop = ga::Population::new(args.population_size, &gen, fitness.clone() );

    let mut generations = 1;
    let mut need_matches = 100;

    for g in 1..args.max_generations {
        if args.verbose {
            println!("{0})", g);
            pop.individuals.iter().take(5).for_each(|ind| {
                println!("\t{0}", ind.fitness)
            });
        }
        if pop.individuals.first().unwrap().fitness >= 1.9 {
            need_matches -= 1;
        } else {
            need_matches = 100;
        }
        if need_matches == 0 {
            break;
        }
        pop = pop.evolve(&gen, fitness.clone());
        generations += 1;
    }

    println!("After {0} Generations", generations);
    pop.individuals.iter().take(5).for_each(|ind| {
        println!("\t{0}", ind.fitness)
    });

    let final_solution = &pop.individuals.first().unwrap().individual;
    println!("Final solution:");
    final_solution.ops.iter().for_each(|op| {
        println!("{0}", op);
    });
}