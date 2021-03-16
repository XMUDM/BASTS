
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

my @ents = $db->ents("c function ~unknown ~unresolved,java method ~abstract,fortran subroutine, fortran function, fortran main program"
                     ." ~unknown ~unresolved,c# method ~abstract,vhdl procedure, vhdl function,ada procedure, ada function");
my $scores = 0;
foreach my $ent (sort{$a->longname() cmp $b->longname();} @ents) {
  # print $ent->freetext();
  if($ent->freetext("CGraph")){
	my $code_id = $ent->name();
	$code_id =~ s/[^0-9]//g;
	my $cfg = $ent->freetext("CGraph");
	open d,">E:\\dataset\\code comment\\new_dataset\\train\\cfgs\\"."$code_id.txt";
	print d $cfg;
	close d;
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