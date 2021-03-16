
# Sample Understand PERL API Script
#
# Synopsis: Lists all functions
#
#   Lists all functions.
#   Requires an existing Understand database
#
# Language: All
#
#  For the latest Understand perl API documentation, see
#      http://www.scitools.com/perl.html
#
#  01-Apr-2013 KG
#
# Usage:
sub usage($) {
    return shift(@_) . <<"END_USAGE";
Usage: $0 -db database
  -db database      Specify Understand database (required for
                    uperl, inherited from Understand)
END_USAGE
}

use Understand;
use Getopt::Long;
use strict;

my $dbPath;
my $help;
GetOptions(
     "db=s" => \$dbPath,
     "help" => \$help,
          );

# help message
die usage("") if ($help);

# open the database
my $db=openDatabase($dbPath);


#code body*******************************************************************

my $scores = 0;

foreach my $file ($db->ents("File")){
  print $file->name()," ";
  foreach my $method ($file->ents("Define","Function")){
    # print $method->name(),"'s metrics:","\n";
    if($method->freetext("CGraph")){
      my $code_id = $file->name();
      $code_id =~ s/[^0-9]//g;
      my $cfg = $method->freetext("CGraph");
      open d,">E:\\chao\\codeSum\\code_process\\train\\cfgs2\\"."$code_id.txt";
      print d $cfg;
      close d;
    }
  }
}


#end body********************************************************************
closeDatabase($db);


# subroutines

sub openDatabase($)
{
    my ($dbPath) = @_;

    my $db = Understand::Gui::db();

    # path not allowed if opened by understand
    if ($db&&$dbPath) {
  die "database already opened by GUI, don't use -db option\n";
    }

    # open database if not already open
    if (!$db) {
  my $status;
  die usage("Error, database not specified\n\n") unless ($dbPath);
  ($db,$status)=Understand::open($dbPath);
  die "Error opening database: ",$status,"\n" if $status;
    }
    return($db);
}

sub closeDatabase($)
{
    my ($db)=@_;

    # close database only if we opened it
    $db->close() if $dbPath;
}