#include "alg.h"

void skip_to_colon( std::ifstream &fin )
{
  char c = ' ';
  while ( c != ':' )
  {
    fin.get( c );
  }
}

void load_settings( isai::ga_settings_t &s )
{
  auto fin = std::ifstream{ "data/config.txt", std::ios::in };

  double dbl;
  std::string str;
  std::size_t szt;

  skip_to_colon( fin );
  fin >> str;
  s.is_input_random = str == "true";

  skip_to_colon( fin );
  for ( auto i = std::size_t{ 0 }; i < 5u; i++ )
  {
    fin >> dbl;
    s.input_coeffs.emplace_back( dbl );
  }

  skip_to_colon( fin );
  fin >> szt;
  assert( szt > 0 );
  s.pop_size = szt;

  skip_to_colon( fin );
  fin >> szt;
  assert( szt > 0 );
  s.max_gens = szt;

  skip_to_colon( fin );
  fin >> szt;
  assert( szt > 0 );
  s.training_data_size = szt;

  skip_to_colon( fin );
  fin >> dbl;
  assert( dbl > 0.0 );
  s.error_threshold = dbl;

  skip_to_colon( fin );
  fin >> dbl;
  assert( dbl >= 0.0 && dbl <= 1.0 );
  s.base_mutation_rate = dbl;
}

void normalize_coeffs( std::vector< double > &coeffs )
{
  for ( auto &&c : coeffs )
  {
    c = std::round( c * 4.0 ) / 4.0;
    if ( c < -15.75 )
    {
      c = -15.75;
    }
    else if ( c > 15.75 )
    {
      c = 15.75;
    }
  }
}


int main( int argc, char *argv[] )
{
  isai::prng_t::initialize();

  auto settings = isai::ga_settings_t{};

  if ( argc == 2 )
  {
    auto param = std::string{ argv[ 1 ] };
    if ( param == "-v" )
    {
      settings.is_verbose = true;
    }
    else
    {
      settings.batch_name = param;
    }
  }
  else if ( argc == 3 )
  {
    auto param1 = std::string{ argv[ 1 ] };
    auto param2 = std::string{ argv[ 2 ] };

    settings.is_verbose = param1 == "-v";
    settings.batch_name = param2;
  }

  load_settings( settings );
  normalize_coeffs( settings.input_coeffs );

  auto ga = isai::genetic_algorithm_t< 35 >{ settings };

  std::puts( "Press any key to run..." );
  std::getchar();

  ga.run();

  auto &&[ res, res_err ] = ga.result();

  std::printf( "Result: " );
  res.print();
  std::printf( "        (i.e. P =" );
  res.print( true );
  std::printf( ")\nError:  %.3f\n", res_err );

  return 0;
}
