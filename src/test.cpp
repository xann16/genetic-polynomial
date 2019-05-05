#include "alg.h"
#include <cstdio>

void basic()
{
  auto ch = isai::chromosome_t< 35 >{};

  std::puts( "GENES:" );
  for ( auto i = std::size_t{ 0 }; i < ch.gene_count(); i++ )
  {
    if ( i % 7 == 0 )
    {
      std::printf( "|" );
    }
    std::printf( "%1d", ch[ i ] ? 1 : 0 );
  }
  std::puts( "|" );

  auto chb = isai::chromosome_t< 35 >{};

  for ( auto i = std::size_t{ 0 }; i < chb.gene_count(); i++ )
  {
    if ( i % 7 == 0 )
    {
      std::printf( "|" );
    }
    std::printf( "%1d", chb[ i ] ? 1 : 0 );
  }
  std::puts( "|" );

  auto poly = isai::to_polynomial( ch );
  auto polyb = isai::to_polynomial( chb );

  std::puts( "\nPOLYNOMIAL COEFFICIENTS:" );
  std::printf( "[ " );
  for ( auto i = std::size_t{ 0 }; i < poly.size(); i++ )
  {
    std::printf( "%5.2f ", poly[ i ] );
  }
  std::printf( "]\n" );

  std::printf( "[ " );
  for ( auto i = std::size_t{ 0 }; i < polyb.size(); i++ )
  {
    std::printf( "%5.2f ", polyb[ i ] );
  }
  std::printf( "]\n" );



  auto data = poly.get_training_data( 10 );
  std::puts( "\nTRAINING DATA:" );
  std::puts( "------------------------------" );
  std::puts( "| lp | X:     | Y:           |" );
  std::puts( "------------------------------" );
  auto index = int{ 1 };
  for ( auto &&p : data )
  {
    std::printf( "| %02d | %+6.2f | %+12.2f |\n", index, p.x, p.y );
    index++;
  }
  std::puts( "------------------------------" );

  std::puts( "FITNESSES:" );
  std::printf( "%10.2f\n", isai::eval_fitness( ch, data ) );
  std::printf( "%10.2f\n", isai::eval_fitness( chb, data ) );
}

int main()
{
  isai::prng_t::initialize();
  // basic();

  // x^2
  // auto poly = isai::polynomial_t< 4 >{ 0.0, 0.0, 1.0, 0.0, 0.0 };
  auto poly = isai::to_polynomial( isai::chromosome_t< 35 >{} );


  std::puts( "GIVEN:" );
  std::printf( "[ " );
  for ( auto i = std::size_t{ 0 }; i < poly.size(); i++ )
  {
    std::printf( "%5.2f ", poly[ i ] );
  }
  std::printf( "]\n" );


  auto td = poly.get_training_data( 50 );
  auto tdcpy = td;

  auto alg = isai::genetic_algorithm_t< 35 >{ 1000, td, 10000, 0.001, 0.01 };

  // alg.print_pop();
  // std::puts( "TRAINING:" );

  alg.train();

  auto chres = alg.get_winner();
  auto res = isai::to_polynomial( chres );

  std::puts( "RESULT:" );
  std::printf( "[ " );
  for ( auto i = std::size_t{ 0 }; i < res.size(); i++ )
  {
    std::printf( "%5.2f ", res[ i ] );
  }
  std::printf( "] | sqerr: %10.2f\n", eval_fitness( chres, tdcpy ) );

  // alg.print_pop();


  return 0;
}
